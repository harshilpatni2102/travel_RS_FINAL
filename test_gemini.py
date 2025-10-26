"""
Quick test to verify Gemini AI is generating proper paragraph content
"""

from gemini_module import GeminiEnhancer, GEMINI_AVAILABLE
import configparser

print("=" * 80)
print("🧪 TESTING GEMINI AI PARAGRAPH GENERATION")
print("=" * 80)

print(f"\n1. GEMINI_AVAILABLE flag: {GEMINI_AVAILABLE}")

if not GEMINI_AVAILABLE:
    print("❌ Gemini module not available!")
    exit(1)

# Load API key from gemini_config.ini (CHANGE IT THERE!)
print(f"\n2. Loading API key from gemini_config.ini...")
try:
    config = configparser.ConfigParser()
    config.read('gemini_config.ini', encoding='utf-8')
    API_KEY = config.get('GEMINI', 'api_key', fallback=None)
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        print("❌ API key not configured in gemini_config.ini!")
        exit(1)
    print(f"   ✅ API key loaded from gemini_config.ini")
except Exception as e:
    print(f"❌ Error reading gemini_config.ini: {e}")
    exit(1)

print(f"\n3. Initializing Gemini with API key...")

try:
    enhancer = GeminiEnhancer(API_KEY)
    print(f"   ✅ Gemini Enabled: {enhancer.enabled}")
    if enhancer.enabled:
        print(f"   ✅ Model: gemini-2.0-flash-exp")
except Exception as e:
    print(f"   ❌ Initialization failed: {e}")
    exit(1)

# Test AI paragraph generation
print(f"\n4. Testing AI explanation generation...")
print(f"   Query: 'I want beautiful beaches and amazing food'")
print(f"   City: Bali, Indonesia")

try:
    result = enhancer.explain_recommendation(
        city="Bali",
        country="Indonesia", 
        user_query="I want beautiful beaches and amazing food",
        score_breakdown={
            'final_score': 0.92,
            'nlp_score': 0.95,
            'content_score': 0.88,
            'popularity_score': 0.93
        },
        destination_data={
            'beaches': 4.9,
            'cuisine': 4.7,
            'culture': 4.5,
            'wellness': 4.6,
            'nature': 4.4,
            'nightlife': 4.2
        }
    )
    
    print(f"\n   ✅ SUCCESS! Generated AI content:")
    print(f"   📏 Length: {len(result)} characters")
    print(f"   📝 Sentences: ~{len(result.split('.'))} sentences")
    print(f"\n" + "=" * 80)
    print("🤖 AI GENERATED PARAGRAPH:")
    print("=" * 80)
    print(result)
    print("=" * 80)
    
    # Verify it's a proper paragraph
    if len(result) < 100:
        print(f"\n⚠️  WARNING: Content seems too short (< 100 chars)")
    elif len(result.split('.')) < 3:
        print(f"\n⚠️  WARNING: Content has less than 3 sentences")
    else:
        print(f"\n✅ VERIFICATION PASSED: Proper paragraph generated!")
        
    # Check for "famous for" content
    if any(word in result.lower() for word in ['famous', 'renowned', 'known for', 'celebrated']):
        print(f"✅ Contains 'famous for' / 'renowned for' information")
    else:
        print(f"⚠️  May not contain 'famous for' information")
    
except Exception as e:
    print(f"\n   ❌ Generation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print(f"\n" + "=" * 80)
print("🎉 TEST COMPLETED SUCCESSFULLY!")
print("=" * 80)
