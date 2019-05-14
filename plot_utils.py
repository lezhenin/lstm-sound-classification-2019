import numpy as np

from matplotlib import pyplot as plt


def plot_confusion_matrix(cm, class_dict, normalize=False, cmap=plt.cm.Blues):
  
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(class_dict))
    plt.xticks(tick_marks, class_dict, rotation=45)
    plt.yticks(tick_marks, class_dict)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    

def plot_history(history):
    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[0].plot(history.history['acc'])
    axs[0].plot(history.history['val_acc'])
    axs[0].set_title('Model accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Test'], loc='upper left')
    axs[0].grid(True)

    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Train', 'Test'], loc='upper left')
    axs[1].grid(True)
    
 
def plot_avg_spectrum(data, labels, class_dict):
    fig, axs = plt.subplots(len(class_dict) // 2 + 1, 2, sharex=True)

    indices = np.arange(len(labels))
    for i in range(len(class_dict)):
        class_indices = list(filter(lambda item: labels[item] == i, indices))
        class_data = data[class_indices]
        avg_spectrum = np.average(class_data, axis=0)
        var_spectrum = np.var(class_data, axis=0)
        axs[i // 2][i % 2].plot(var_spectrum)
        axs[i // 2][i % 2].plot(avg_spectrum)
        axs[i // 2][i % 2].set_title(class_dict[i])
    avg_spectrum = np.average(data, axis=0)
    var_spectrum = np.var(data, axis=0)

    axs[-1][0].plot(avg_spectrum)
    axs[-1][0].plot(var_spectrum)
    axs[-1][0].set_title('general')
