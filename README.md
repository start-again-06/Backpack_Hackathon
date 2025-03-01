✨ Tabular Playground Series ✨

Welcome to the Tabular Playground Series! This repository contains scripts and resources for tackling Kaggle's Tabular Playground competitions, designed to help the community learn and sharpen their data science skills.

🌟 Overview

The Tabular Playground Series provides lightweight machine learning challenges that:

⏳ Run for a few weeks.

📚 Use synthetic datasets based on real-world data.

🎯 Help improve modeling and feature engineering skills.

🎨 Encourage creativity in data visualization.

💡 Why Synthetic Data?

Synthetic datasets allow for more interesting challenges while keeping test labels private. Over time, the quality of synthetic data has improved significantly, leading to:

✅ More realistic patterns.

⚡ Reduced data leakage risk.

✉ Opportunities for feedback and continuous improvement.

👨‍💻 Dependencies

To get started, install the necessary Python libraries: pip install numpy pandas matplotlib seaborn missingno scikit-learn tensorflow cudf cuml

📊 Exploratory Data Analysis (EDA)

The repository includes detailed EDA covering:

📅 Summary statistics of train, test, and extra training datasets.

📈 Price distribution visualization.

🔄 Comparison of numerical and categorical variables.

🌐 Missing values detection and imputation.

⚖️ Feature encoding and scaling.

🌐 Data Preprocessing

⚖️ Target Encoding: Applied on categorical features.

🏢 Standard Scaling: Normalizing numeric features.

💪 Handling Missing Values: Using median imputation and 'None' for categorical variables.

🤖 Deep Neural Network (DNN) Model

A simple yet effective Deep Learning Model using TensorFlow/Keras:

🛠️ Three dense layers with Batch Normalization and Dropout.

⚙️ L2 regularization to prevent overfitting.

⏳ Early Stopping to optimize training.

🌍 RMSE tracking for model performance.

🎉 Results & Visualization

🌟 Validation RMSE: Tracked during training.

💡 Heatmaps & Boxplots: Feature correlations and distributions.

✨ Scatter Plots: Actual vs Predicted values.

💾 Submission

Once the model is trained:

submission.to_csv('sample_submission.csv', index=False)

Download and submit predictions to Kaggle!

📍 Repository Structure

|-- data/ # Raw and processed datasets |-- notebooks/ # Jupyter notebooks for analysis |-- models/ # Trained model files |-- scripts/ # Python scripts for preprocessing & modeling |-- README.md # Project documentation 💡 Contributions

Want to improve this project? Feel free to fork, star, and submit PRs!

✨ Stay Connected

🌟 Follow the Kaggle Competition here

👤 Connect with us on LinkedIn/Twitter!

Happy Coding! 💪
