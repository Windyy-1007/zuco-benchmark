# Algorithm and Mathematical Measurements for EEG Performance Analysis and Dyslexia Labeling

This document outlines the algorithm and mathematical measurements used to analyze EEG performance and label subjects as dyslexic or normal.

---

## **Algorithm for Analyzing EEG Performance**

### **1. Load EEG Features**
- For each subject, load EEG electrode features stored in `.npy` files.
- Each subject's data contains multiple samples, where each sample includes features and a label.

### **2. Extract Features**
- For each sample, separate the features (all elements except the last) and the label (last element).

### **3. Compute Performance Metrics**
For each subject, compute the following metrics:

1. **Mean Signal Strength**:
   \[
   \text{mean\_signal} = \text{mean}(|\text{features\_array}|)
   \]
   Measures the average absolute signal strength across all electrodes and samples.

2. **Signal Variance**:
   \[
   \text{signal\_variance} = \text{var}(\text{features\_array})
   \]
   Indicates the variability of the signal, with higher values suggesting more active brain activity.

3. **Signal Power**:
   \[
   \text{signal\_power} = \text{mean}(\text{features\_array}^2)
   \]
   Represents the average power of the signal.

4. **Number of Samples**:
   \[
   n\_samples = \text{len}(\text{features\_list})
   \]
   Reflects the reading efficiency of the subject.

5. **Feature Stability**:
   \[
   \text{feature\_stability} = \frac{1}{1 + \text{mean}(\text{std}(\text{features\_array}, \text{axis}=0))}
   \]
   Measures the consistency of features across samples, with lower standard deviation indicating higher stability.

### **4. Compute Composite EEG Performance Score**
Combine the above metrics into a single score using weighted contributions:
\[
\text{performance\_score} = (\text{signal\_variance} \cdot 0.3) + (\text{mean\_signal} \cdot 0.25) + (\text{signal\_power} \cdot 0.2) + \left(\frac{n\_samples}{100} \cdot 0.1\right) + (\text{feature\_stability} \cdot 0.15)
\]
- A lower score indicates poorer neural efficiency, which may suggest dyslexia.

### **5. Rank Subjects**
- Sort subjects by their performance scores in ascending order.
- The three subjects with the lowest scores are labeled as dyslexic.

---

## **Mathematical Measurements**

1. **Mean Signal Strength**:
   - Captures the average magnitude of the EEG signal.

2. **Signal Variance**:
   - Quantifies the spread of the signal values, indicating variability in brain activity.

3. **Signal Power**:
   - Represents the energy of the signal, useful for assessing overall activity levels.

4. **Number of Samples**:
   - Indicates the amount of data collected, which can correlate with reading efficiency.

5. **Feature Stability**:
   - Measures the consistency of EEG features across samples, with higher stability suggesting better neural efficiency.

6. **Composite Score**:
   - A weighted combination of the above metrics to provide an overall measure of EEG performance.

---

## **Labeling Algorithm**

1. **Sort Subjects**:
   - Rank subjects based on their composite performance scores.

2. **Assign Dyslexia Labels**:
   - The three subjects with the lowest scores are labeled as dyslexic (`1`).
   - All other subjects are labeled as normal (`0`).

---

## **Visualization and Validation**

1. **Bar Plot**:
   - Visualize the performance scores of all subjects, with dyslexic subjects highlighted in red.

2. **Box Plot**:
   - Compare the distribution of performance scores between dyslexic and normal groups.

---

This approach combines statistical analysis, signal processing, and machine learning-inspired metrics to identify dyslexic candidates based on EEG data.
