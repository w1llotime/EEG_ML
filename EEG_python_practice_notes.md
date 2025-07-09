# EEG ML Practice Model
This is a basic EEG signal generation + machine learning model, only for learning purposes. This signal is not a real signal, rather generated using the superposition of sine waves. 
Most of the code is from gemini and I've only tried to replicate it. The matplotlib part is entirely copied from gemini.

After running it you can see the following nice diagram.

<img width="400" alt="Screenshot 2025-07-09 at 4 15 38 PM" src="https://github.com/user-attachments/assets/82e5b07f-1274-4159-adc6-eedb67e2ea38" />

Inside this file I will be discussing how each step works and what I learned. I obviously haven't fully understood it, but I'm glad I made this rough model which I built an understanding and can later implement it fully on my own and improve on it.

# Steps
## Step 1 - EEG signal generation
1. We need to generate a EEG signal using sine waves. First we define the time range where the signal is recorded. It involves a starting point, endpoint, and the amount of timestamps within the signal.
2. We need to define dominant frequencies, e.g. [6Hz, 8Hz, 15Hz] for our signal classes. We treat each frequency like a constant in the sine function, in which time is an input that is varied along the range defined in the first step. And thus we get a different sine wave for each frequency.
3. Lastly we superimpose the waves on each other to get a mixed signal.


```python
import numpy as np
from scipy.signal import welch # More robust for power spectral density
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


def generate_eeg (total_duration, dominant_freq, sfreq, noise = 0.5):
    time = np.linspace(0, total_duration, int(total_duration * sfreq))
    signal = np.zeros_like(time) 
    for freq in (dominant_freq):
        signal += np.sin(2 * np.pi * time * freq) 
    signal += np.random.randn(len(time))
    return signal

total_duration = 2 # 100 seconds
s_freq = 250 # 10 s-1
dominant_freq_class0 = [5,8,10] #dominant frequency 0
dominant_freq_class1 = [15,20,30] #dominant frequency 1
signal0 = generate_eeg (total_duration, dominant_freq_class0, s_freq, noise = 0.5) #the signal for class 0
signal1 = generate_eeg (total_duration, dominant_freq_class0, s_freq, noise = 0.5) # signal for class 1
n_epochs_per_class = 200 # Number of simulated trials for each class

eeg_data = []
labels = []

for _ in range (n_epochs_per_class):
    eeg_data.append(signal0)
    labels.append(0)
# Note that we have defined epochs to be 200. Epochs are samples of EEG data.

for _ in range (n_epochs_per_class):
    eeg_data.append(signal1)
    labels.append(1)
```
Up to now, eeg_data is a list of epochs that contain different values. Note that each epoch represents a superimposed sine wave. Each epoch is a parenthesis that a sequence of values that are the amplitudes of a mixed wave. So the eeg_data is a list that contains many of those epochs, separated by commas. But after this line, down below, we convert them into an np array, and every comma is now the start of a new line, and it becomes a 2D numpy array where the number of rows equal the number of epochs, where each column represents each amplitude contained in each epoch. Now if each epoch has a sampling frequency of 250 Hz, and a duration of 2 seconds, then we have 500 amplitudes.
```python
eeg_data = np.array(eeg_data)
# We can see that EEG data here is a mix of class 0 and class 1. This is done because we later need to classiy test data into either class 0 or class 1. The mix of EEG signals here have 400 rows. Because eeg_data.append(signal0) added 200 rows of EEG signals in the form of np arrays.

labels =np.array(labels)
```
## Step 2. Feature extraction from data
What are features in terms of eeg data? Well, it is pretty similar to features in other data. It represents input variables that give us insight about the output. We describe each feature's significance through the weight and biases we assign to them. We optimize the prediction model by changing these parameters, thereby reducing the error to the true data.

If we have 4 types of frequency bands (e.g. delta, alpha, beta, theta), our features would be a matrix composed of n rows and m columns. n is the number of epochs, training samples that each represent a unique window of time of our EEG recording. m columns represent the number of frequencies we care about. If we have 4 frequencies, we'd have 4 columns. This is because each feature[i] here for each row we have represents the amount of power each frequency contributes inside a sample! The feature for one row would be [Delta_Power, Theta_Power, Alpha_Power, Beta_Power]. This gives a lot of insight about the composition of each class and helps us predict new inputs of EEG data with different powers, and then we can classify it logistically.

Recall that each epoch consists of superpositions of sine waves with different frequencies. That's why we need to extract out the strengths and contributions of each frequency wave to the overall epoch. This helps us classify whether the epoch is in class0 or class1.

```python

print (eeg_data.shape) #This should give 400 epochs (rows) times the number of time stamps in an epoch (columns)
print (labels.shape)

def extract_features (total_epochs, sfreq): #sfreq is always sampling frequency
    features =[]
    
    bands = {"delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),}
    for i, epoch in enumerate (total_epochs):
        nperseg = min(sfreq * 2, len(epoch))
        if nperseg == 0: # Handle very short epochs if they occur
            print(f"Warning: Epoch {i} is too short to compute PSD. Skipping.")
            features.append(np.zeros(len(bands))) # Append zeros if epoch too short
            continue
        #Welch's method works by dividing a signal into shorter, overlapping segments. Nperseg tells the welch function below how many                  amplitude samples to include for each epoch. We do not include every sample (that might be thousands). Instead, we include some, and            then average them. 
        
        band_powers = []   
        freqs, psd = welch(epoch, sfreq, nperseg=nperseg, noverlap=nperseg // 2)
        # this line needs to be in the loop. For we need to do frequencies and psd for each epoch.
        # The welch function gives the entire frequency spectrum that can be found in our epoch. It doesn't give only the dominant frequencies we used to make our EEG signals, but they will show up as prominent peaks inside it. psd is the power spectral density value associated with each frequency value that the welch function finds.

        for band_name, (low, high) in bands.items():
            #by calling bands.items(), we are getting the name of the band(alpha, as well as its min and max values in parenthesis (0.5, 4.0)), and so this loop is running through each band, checking if each of the frequencies in our frequencies (broken down and extracted through the welch method are inside that range. If it is in that range, the frequency is true in that index, and false otherwise. So the index here is a boolean array that consists of only true or false. Each of the true and false are corresponded with each epochs' frequenies. 
            index = np.logical_and(low <= freqs, freqs <= high)
            # now let's integrate to find the power contained in each band
            power = np.trapezoid(psd[index], freqs[index])
            # for each band, we are calculating the power of that frequency. When we index by using psd[index] and freqs[index], the value in each array that corresponds to "true" would be called. 
            band_powers.append(power)
        features.append(band_powers) #by the end, we have a list of powers! Each row is an epoch. And the values in each row represent the amount of power each frequency contributes. This is because we are running thorugh each of the frequencies. 


#But how does the above code block transform the wide range of frequency values to 4 distinct values on each row, representing the amount of strength of each frequency band?? Let's examine.
#power = np.trapezoid(psd[idx_band], freqs[idx_band]) is the key step that does it. As discussed above, we are only selecting psd and frequency values that returned true for the frequency band. np.trapezoid performs integration of the power spectral density with respect to the frequency bands selected. It gets the area under that psd curve, obtaining total power contained inside those frequencies. That results in a single value, representing the total power, for example, for the delta band. band_powers.append(power). We will be appending that value to the band_powers. Since we are iterating for each band name, we will have a delta value, alpha value, beta value, and theta value for the epoch we are looking at right now. This is for the inner loop. But since our outer loop is iterating through each epoch, we will get those 4 band power values appended to the features for each epoch. This results in a features matrix, with each row being our epoch, and each element in each row being the 4 band power strengths found inside that epoch!

    return np.array(features)
    
# Now we have to separate the model into training data and testing data.

X= extract_features(eeg_data, s_freq) #we get features for our data.
y= labels
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# test_size=0.3:
# This parameter tells train_test_split to allocate 30% of the total data to the testing set and the remaining 70% to the training set.
# For the testing set: 0.3 * 400 = 120 samples (epochs)
# For the training set: 0.7 * 400 = 280 samples (epochs)

print(f"Training features shape: {x_train.shape}")
print(f"Testing features shape: {x_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Testing labels shape: {y_test.shape}")

# --- 4. Build and Train a Classification Model ---
# We'll use Logistic Regression, a simple yet effective linear model.

model = LogisticRegression(max_iter=1000) # Increase max_iter for convergence
model.fit(x_train, y_train)

# --- 5. Evaluate the Model ---
print("Evaluating model performance...")
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Relaxed (0)", "Active (1)"])

print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)

# --- 6. Visualization (Optional) ---
# Plot an example of raw signal and its PSD for each class

plt.figure(figsize=(14, 8))

# Plot raw signal examples
plt.subplot(2, 2, 1)
plt.plot(np.linspace(0, total_duration, eeg_data.shape[1], endpoint=False), eeg_data[0])
plt.title('Example Raw EEG Signal - Class 0 (Relaxed)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(np.linspace(0, total_duration, eeg_data.shape[1], endpoint=False), eeg_data[n_epochs_per_class])
plt.title('Example Raw EEG Signal - Class 1 (Active)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot PSD examples
# Re-calculate PSD for plotting (already done for features, but good to visualize)
freqs0, psd0 = welch(eeg_data[0], s_freq, nperseg=min(s_freq * 2, eeg_data.shape[1]), noverlap=min(s_freq * 2, eeg_data.shape[1]) // 2)
freqs1, psd1 = welch(eeg_data[n_epochs_per_class], s_freq, nperseg=min(s_freq * 2, eeg_data.shape[1]), noverlap=min(s_freq * 2, eeg_data.shape[1]) // 2)

plt.subplot(2, 2, 3)
plt.semilogy(freqs0, psd0)
plt.title('PSD for Class 0 (Relaxed)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (V^2/Hz)')
plt.xlim(0, 50) # Limit x-axis for better visualization of main bands
plt.grid(True)

plt.subplot(2, 2, 4)
plt.semilogy(freqs1, psd1)
plt.title('PSD for Class 1 (Active)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (V^2/Hz)')
plt.xlim(0, 50) # Limit x-axis for better visualization of main bands
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n--- Model Coefficients (Feature Importance) ---")
# The coefficients of the Logistic Regression model can give insight into
# which frequency bands were most important for classification.
# The order of coefficients corresponds to the order of bands:
# [delta, theta, alpha, beta, gamma]
band_names = ["delta", "theta", "alpha", "beta"]
for i, coef in enumerate(model.coef_[0]):
    print(f"{band_names[i]}: {coef:.4f}")

# Interpretation: A positive coefficient means that higher power in that band
# is associated with the positive class (Class 1 - "Active" in our case).
# A negative coefficient means higher power is associated with the negative
# class (Class 0 - "Relaxed").

```
A sample output for this part at the end is:

--- Model Coefficients (Feature Importance) ---
delta: -0.3661
theta: 4.5191
alpha: -6.8821
beta: 1.3479

There are like the m1, m2, m3, m4 we commonly see in ML models. They are the weights that describe how much each input is weighed into determining the class of the sample input. Thus, we ahve 4 inputs here for each epoch, and each input is the the total power of the frequency band it corresponds to. e.g. the first value is the total power of the delta band for that epoch. Because it is a good quantifiable way to measure the contribution of each band.




