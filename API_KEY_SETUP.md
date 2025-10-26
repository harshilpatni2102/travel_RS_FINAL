# 🔑 How to Change Your Gemini API Key

## ⚡ Quick Instructions (SUPER EASY!)

Your Gemini API key is stored in **ONE FILE**: `gemini_config.ini`

### To Change Your API Key:

1. **Open the file:** `gemini_config.ini` (in the project root folder)

2. **Find this line:**
```ini
api_key = AIzaSyBT6kVFVOcUn5J1WFp3sduj29ufm5tU5kA
```

3. **Replace with your new key:**
```ini
api_key = YOUR_NEW_API_KEY_HERE
```

4. **Save the file** - That's it! No restart needed for new sessions.

---

## 📍 Exact Location

```
rs_travel/
  ├── app.py
  ├── gemini_config.ini  ← CHANGE YOUR API KEY HERE! 🔑
  ├── config.ini
  └── ...
```

Open `gemini_config.ini` and you'll see:
```ini
[GEMINI]
api_key = YOUR_KEY_HERE  ← Change this line
```

---

## Getting a New API Key

If you need a new Gemini API key:

1. **Go to:** https://aistudio.google.com/apikey
2. **Sign in** with your Google account
3. **Click** "Create API Key"
4. **Copy** the key
5. **Paste** it in `gemini_config.ini` (replace the old one)

---

## Quota Limits

**Free Tier:**
- ✅ 50 requests per day
- 🔄 Resets every 24 hours

**If you hit the limit:**
- ⏰ Wait until tomorrow for quota reset
- 💰 OR upgrade to a paid plan for higher limits

---

## Testing Your New Key

After changing the API key in `gemini_config.ini`:

```bash
python test_gemini.py
```

This will verify:
- ✅ Key loads correctly
- ✅ Gemini API is working  
- ✅ AI paragraph generation works

---

## Summary

🔑 **Change your API key here:** `gemini_config.ini` (line 9)

That's the ONLY place you need to change it!

---

**Last Updated:** October 27, 2025 @ 4:30 AM 😊
