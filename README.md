Support Vector Machines (SVMs) finding a boundary that best separates data points of different classes, creating a clear distinction between them. A hyperplane is a decision boundary that divides the feature space into two distinct classes (Classes means Target values if want to know the Good days and bad days of the stocks, in this there are 2 classes). 
Sometimes, a straight hyperplane may not work. For example, suppose you also consider features like sentiment or market trend along with Volume and Volatility. The data might not be separable by a simple linear line.
Here, SVC uses a kernel to transform the data into a higher-dimensional space, where it can find a linear separator