# 🚀 Quick Start Guide

## Travel Destination Recommendation System

### ⚡ 3-Step Setup

#### Step 1: Run Setup
Double-click `setup.bat` or run in terminal:
```bash
setup.bat
```

#### Step 2: Launch Application
Double-click `run_app.bat` or run:
```bash
run_app.bat
```

#### Step 3: Open Browser
Application will automatically open at: `http://localhost:8501`

---

## 📝 Manual Setup (Alternative)

If batch files don't work:

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run application
streamlit run app.py
```

---

## 💡 How to Use

### Get Recommendations

1. **Enter your travel preferences** in the sidebar:
   - Example: "I love beaches and seafood in tropical destinations"

2. **Apply filters** (optional):
   - Select region (Europe, Asia, etc.)
   - Choose budget level
   - Set minimum rating

3. **Click "Get Recommendations"**

4. **Explore results**:
   - View personalized destination cards
   - See score breakdowns
   - Download recommendations as CSV

### Explore Other Features

- **Popular Destinations**: Browse top-rated places
- **Visualizations**: Interactive charts and world map
- **Search**: Look up specific cities

---

## 🔧 Troubleshooting

### Problem: "Python is not recognized"
**Solution**: Install Python 3.9+ from https://www.python.org/

### Problem: "Module not found" error
**Solution**: Ensure virtual environment is activated and run:
```bash
pip install -r requirements.txt
```

### Problem: "Dataset not found"
**Solution**: Verify `Worldwide-Travel-Cities-Dataset-Ratings-and-Climate.csv` is in `data/` folder

### Problem: Application won't start
**Solution**: 
1. Check if port 8501 is available
2. Try: `streamlit run app.py --server.port 8502`

### Problem: Slow first load
**Solution**: Normal! The BERT model is downloading (500MB). Subsequent loads are fast.

---

## 📊 System Requirements

- **Python**: 3.9 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: 2GB free space
- **Internet**: Required for first-time model download
- **Browser**: Chrome, Firefox, Safari, or Edge

---

## 🎓 For Academic Submission

### Files to Submit

1. **Source Code**: All `.py` files
2. **Data**: CSV file in `data/` folder
3. **Documentation**: 
   - `README.md`
   - `ACADEMIC_DOCUMENTATION.md`
4. **Dependencies**: `requirements.txt`
5. **License**: `LICENSE`

### Presentation Tips

1. **Demo the live application** - most impressive!
2. **Show code architecture** - highlight modularity
3. **Explain NLP pipeline** - preprocessing → embeddings → similarity
4. **Demonstrate hybrid recommender** - show score breakdown
5. **Present visualizations** - charts and interactive map

### Key Points to Emphasize

- ✅ **NLP**: Uses state-of-the-art BERT embeddings
- ✅ **Recommender**: Hybrid approach with explainability
- ✅ **Engineering**: Production-ready, modular code
- ✅ **UX**: Interactive, user-friendly interface
- ✅ **Innovation**: Novel integration of both technologies

---

## 🐛 Known Limitations

1. **Language**: Currently English only
2. **Data**: Static dataset (no real-time updates)
3. **Scale**: Optimized for single user (can be scaled)
4. **Personalization**: No user accounts or history tracking

---

## 📞 Support

For help or questions:

1. Check `README.md` for detailed documentation
2. Review `ACADEMIC_DOCUMENTATION.md` for technical details
3. Examine inline code comments
4. Test individual modules: `python nlp_module.py`

---

## ✅ Success Checklist

Before submitting or presenting:

- [ ] Application runs without errors
- [ ] All dependencies installed correctly
- [ ] Dataset file in correct location
- [ ] Can generate recommendations successfully
- [ ] All visualizations working
- [ ] Search functionality operational
- [ ] Code is well-commented
- [ ] Documentation is complete
- [ ] GitHub repository (optional but recommended)

---

**Ready to go! Good luck with your submission! 🎉**

---

*For detailed technical information, see `README.md` and `ACADEMIC_DOCUMENTATION.md`*
