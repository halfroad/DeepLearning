import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def Prepare():

    df = pd.read_csv("../../../../MyBook/Chapter-6-Spam-Email-Prediction/spam_emails.csv")

    print(df.head())

    spamEmails = df[df["spam"] == 1]
    notSpamEmails = df[df["spam"] == 0]

    print("There are {} emails, normal emails are {}, spam emails are {}.".format(df.shape, len(notSpamEmails), len(spamEmails)))

    # Set the default style for plot
    plt.style.use("default")

    grouped = df.groupby(["spam"]).count()
    print(grouped.head())

    X = []

    X.append("Not Spam" if grouped.index.values[0] == 0 else "")
    X.append("Spam" if grouped.index.values[1] == 1 else "")

    # Or use the element name as the label
    y = grouped.values[:, 0]

    # Create a figure object
    figure, ax = plt.subplots()

    # Set the size of figure
    figure.set_size_inches(10, 5)

    # Draw the bar figure
    ax.bar(X, y)

    # Set the title
    plt.title("Spam Email Bar Graph")

    # Set the title for axis Y
    plt.ylabel("Number of Emails")

    # Show the grid
    plt.grid()

    # show the plot
    plt.show()

    df["text"] = [text[len("Subject: "):] for text in df["text"]]
    df["length"] = df["text"].map(lambda text: len(text))

    print(df.head())

    print(df.isnull().sum())
    
    return df

def PlotBasicEmails(spamEmailLengths, normalEmailLengths):
    
    figure, ax = plt.subplots()
    
    figure.set_size_inches(10, 5)
    
    # Plot the spam emails
    ax.plot(np.array(range(spamEmailLengths.shape[0])), np.sort(spamEmailLengths), color = "r", alpha = 0.5, label = "Spam")
    # Plot the normal emails
    ax.plot(np.array(range(normalEmailLengths.shape[0])), np.sort(normalEmailLengths), color = "g", alpha = 0.5, label = "Not Spam")
    
    plt.title("Spam Emails and Normal Emails Distribution Diagram")
    
    # Legend is used to set the figure
    # loc = 1 means the legend will be shown on top right, size is to set font
    plt.legend(loc = 1, prop = {"size": 12})
    
    # Show the grid
    plt.grid()
    plt.show()


df = Prepare()

# Grab all the lengths of text in spam emails
spamEmailLengths = df[df["spam"] == 1]["length"].values
# Grab all the lengths of text in normal emails
normalEmailLengths = df[df["spam"] == 0]["length"].values

PlotBasicEmails(spamEmailLengths, normalEmailLengths)