# EU Car Price Predictor v2.0 🚗 AI-Enhanced

An advanced automotive market analysis tool developed as a collaborative effort between **Human Intelligence** and **AI (Gemini)**. This project is a major evolution of previous price prediction models, offering higher precision and more robust features.

## 🌟 Key Improvements over v1.0
Compared to its predecessors, this version introduces significant upgrades:
* **Larger Dataset:** Trained on a substantially expanded dataset with more diverse European market listings.
* **Two-Stage XGBoost Architecture:** Utilizes a specialized dual-model approach (Technical Specs + Brand Influence) for superior accuracy.
* **Automated Feature Encoding:** Integrated `mappings.json` system that eliminates manual data entry errors by automatically handling brand and model encoding.
* **Smart Power Estimation:** Built-in logic to estimate Kilowatts (kW) and Horsepower (HP) based on engine capacity.
* **Market Range Prediction:** Instead of a single number, the model provides a **+/- €2,000 Market Band** to reflect real-world price fluctuations.

## 🛠 Features
- **Interactive CLI:** A user-friendly command-line interface for real-time predictions.
- **Euro-Centric Logic:** Fully optimized for European currency and metrics.
- **Feature Engineering:** Includes advanced features like car age (at the time of listing), mileage density, and brand prestige factors.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- XGBoost
- Pandas / Numpy

### Installation & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/K-David-Ai-Dev/Eu-Car-Predict.git](https://github.com/K-David-Ai-Dev/Eu-Car-Predict.git)
Run the predictor:

Bash
python predict.py
🤖 Collaborative Development
This project is a testament to modern software development, where human domain expertise meets AI's analytical power. All logic, from the mappings.json automation to the predictive algorithms, was refined through an iterative collaborative process.

Developed by K-David & Gemini (Lumina)
This project is a testament to modern software development, where human domain expertise meets AI's analytical power. All logic, from the mappings.json automation to the predictive algorithms, was refined through an iterative collaborative process.

Developed by K-David & Gemini (Lumina)
