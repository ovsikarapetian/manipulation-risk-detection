import librosa
import numpy as np
import joblib
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

CONFIDENCE_THRESHOLD = 0.65 
HIGH_CONFIDENCE_THRESHOLD = 0.85  

def extract_features(file_path):
    """Հատկանիշների արդյունահանում ձայնային ֆայլից"""
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        
        
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
       
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        energy = np.mean(librosa.feature.rms(y=audio))
        
        return {
            'mfccs': mfccs_mean,
            'spectral_centroid': spectral_centroid,
            'zcr': zcr,
            'energy': energy
        }
    except Exception as e:
        print(f"Սխալ՝ {e}")
        return None

def detect_manipulation(emotion, confidence, all_probabilities, extra_features):
    """
    Մանիպուլյացիոն հարձակումների հայտնաբերում
    
    Վերադարձնում է dict՝ manipulation_detected (bool), risk_level, warnings
    """
    warnings_list = []
    risk_level = "ՑԱԾՐ"
    manipulation_detected = False
    
    if confidence is None:
        warnings_list.append("⚠️ Վստահության աստիճանը հասանելի չէ (մոդելը վերապատրաստել է անհրաժեշտ)")
        return {
            'manipulation_detected': False,
            'risk_level': "ԱՆՈՐՈՇ",
            'warnings': warnings_list
        }
    
    if confidence < CONFIDENCE_THRESHOLD:
        warnings_list.append(f"⚠️ Ցածր վստահության աստիճան ({confidence:.2%}) - հնարավոր է անորոշ կամ խառն էմոցիա")
        risk_level = "ՄԻՋԻՆ"
        manipulation_detected = True
    
    if emotion == "happy" and 0.4 <= confidence <= 0.7:
        warnings_list.append("🎭 ԿԱՍԿԱԾԵԼԻ: Հնարավոր կեղծ դրական էմոցիա (ժպիտի մանիպուլյացիա)")
        risk_level = "ԲԱՐՁՐ"
        manipulation_detected = True
    
    
    if all_probabilities is not None:
        emotion_probs = dict(zip(
            ["angry", "fear", "happy", "neutral", "sad"],
            all_probabilities[0]
        ))
        
        if emotion == "neutral" and emotion_probs.get("angry", 0) > 0.25:
            warnings_list.append("😐➡️😠 ԿԱՍԿԱԾԵԼԻ: Չեզոք տոն բայց բարձր բարկության հավանականություն")
            risk_level = "ԲԱՐՁՐ"
            manipulation_detected = True
    
    if extra_features:
        energy = extra_features.get('energy', 0)
        if energy > 0.15:  # Շատ բարձր էներգիա չեզոք էմոցիայի համար
            if emotion in ["neutral", "happy"]:
                warnings_list.append("📊 Անսովոր էներգետիկ մակարդակ տվյալ էմոցիայի համար")
                risk_level = "ՄԻՋԻՆ"
    
    if confidence > HIGH_CONFIDENCE_THRESHOLD and emotion == "fear":
        warnings_list.append("😨 Հստակ վախի ազդանշան - հնարավոր սպառնալիք կամ ճնշում")
        risk_level = "ԲԱՐՁՐ"
        manipulation_detected = True
    
    return {
        'manipulation_detected': manipulation_detected,
        'risk_level': risk_level,
        'warnings': warnings_list
    }

def predict_emotion_with_analysis(file_path, show_probabilities=True):
    """
    Էմոցիայի կանխատեսում + մանիպուլյացիայի հայտնաբերում
    """
    print("\n" + "="*60)
    print(f"🎤 Վերլուծվում է՝ '{file_path}'")
    print(f"⏰ Ժամանակ՝ {datetime.now().strftime('%H:%M:%S')}")
    print("="*60 + "\n")
    
    try:
        features_dict = extract_features(file_path)
        if features_dict is None:
            return
        
        mfccs = features_dict['mfccs']
        
        model = joblib.load('emotion_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        features_2d = mfccs.reshape(1, -1)
        features_scaled = scaler.transform(features_2d)
        
        prediction = model.predict(features_scaled)
        emotion = prediction[0]
        
        probabilities = None
        confidence = None
        
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features_scaled)
                confidence = np.max(probabilities)
            else:
                print("⚠️ ՆՇՈՒՄ: Մոդելը չունի հավանականության հաշվարկման հնարավորություն")
                print("   Վերապատրաստեք մոդելը probability=True պարամետրով SVC-ում\n")
        except Exception as e:
            print(f"⚠️ Հավանականությունների հաշվարկման սխալ՝ {e}\n")
        
        print("📊 ՎԵՐԼՈՒԾՈՒԹՅԱՆ ԱՐԴՅՈՒՆՔՆԵՐ:")
        print("-" * 60)
        print(f"🎯 Հայտնաբերված էմոցիա՝    {emotion.upper()}")
        
        if confidence is not None:
            print(f"💯 Վստահության աստիճան՝     {confidence:.2%}")
            
            if confidence >= HIGH_CONFIDENCE_THRESHOLD:
                conf_assessment = "🟢 Շատ բարձր"
            elif confidence >= CONFIDENCE_THRESHOLD:
                conf_assessment = "🟡 Լավ"
            else:
                conf_assessment = "🔴 Ցածր"
            print(f"📈 Գնահատում՝                {conf_assessment}")
        else:
            print(f"💯 Վստահության աստիճան՝     N/A (վերապատրաստեք մոդելը)")
        
        if show_probabilities and probabilities is not None:
            print("\n📋 ԲՈԼՈՐ ԷՄՈՑԻԱՆԵՐԻ ՀԱՎԱՆԱԿԱՆՈՒԹՅՈՒՆՆԵՐԸ:")
            print("-" * 60)
            emotion_labels = model.classes_
            for i, (emo, prob) in enumerate(zip(emotion_labels, probabilities[0])):
                bar_length = int(prob * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                marker = " ← ԸՆՏՐՎԱԾ" if emo == emotion else ""
                print(f"{emo:10s} │ {bar} │ {prob:6.2%}{marker}")
        
        print("\n" + "="*60)
        print("🔍 ՄԱՆԻՊՈՒԼՅԱՑԻԱՅԻ ՀԱՅՏՆԱԲԵՐՄԱՆ ՎԵՐԼՈՒԾՈՒԹՅՈՒՆ")
        print("="*60)
        
        manipulation_result = detect_manipulation(
            emotion, 
            confidence, 
            probabilities,
            features_dict
        )
        
        if manipulation_result['manipulation_detected']:
            print(f"\n⚡ ԿԱՍԿԱԾԵԼԻ ԱԿՏԻՎՈՒԹՅՈՒՆ ՀԱՅՏՆԱԲԵՐՎԱԾ")
            print(f"🎚️  Ռիսկի մակարդակ՝ {manipulation_result['risk_level']}")
            print(f"\n📝 Նախազգուշացումներ՝")
            for warning in manipulation_result['warnings']:
                print(f"   • {warning}")
        else:
            print("\n✅ Մանիպուլյացիայի հայտեր չհայտնաբերվեցին")
            if confidence is not None:
                print("✓ Էմոցիան հավաստի է և բնական")
        
        print("\n" + "="*60)
        print("🔧 ԼՐԱՑՈՒՑԻՉ ՏԵԽՆԻԿԱԿԱՆ ՏՎՅԱԼՆԵՐ")
        print("="*60)
        print(f"🎵 Սպեկտրալ կենտրոն՝      {features_dict['spectral_centroid']:.2f} Hz")
        print(f"〰️  Զրոյական անցումներ՝    {features_dict['zcr']:.4f}")
        print(f"⚡ Էներգիա՝                {features_dict['energy']:.4f}")
        
        print("\n" + "="*60 + "\n")
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': probabilities,
            'manipulation': manipulation_result,
            'features': features_dict
        }
        
    except FileNotFoundError:
        print(f"❌ Ֆայլը չի գտնվել՝ {file_path}")
        print("   Ստուգեք ֆայլի անունը և դրա գոյությունը։")
    except Exception as e:
        print(f"❌ Սխալ՝ {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_files = [
          "test_voice1.wav",
        # "my_voice1.wav",
        # "my_voice2.wav",
        # "my_voice3.wav",
    ]
    
    for test_file in test_files:
        result = predict_emotion_with_analysis(test_file, show_probabilities=True)
        if result:
            pass
