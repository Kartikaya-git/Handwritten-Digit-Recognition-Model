import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import os

cwd = os.getcwd()
new_folder = "/Users/kartikayasrivastava/Desktop/Digit_24oct"  
os.chdir(new_folder)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten images from (28,28) → (784,)
x_train=x_train.reshape(60000,784)/255.0
x_test=x_test.reshape(10000,784)/255.0

# One-hot encode labels
def one_hot(y):
    one_hot_y=np.zeros((y.size, 10))
    one_hot_y[np.arange(y.size), y]=1
    return one_hot_y.T # Shape: (10, number of samples)
y_train_enc=one_hot(y_train)
y_test_enc=one_hot(y_test)

def init_params():
    w1=np.random.rand(128,784)*0.01
    b1=np.zeros((128,1))
    w2=np.random.rand(10,128)*0.01
    b2=np.zeros((10,1))
    return w1,b1,w2,b2

def relu(Z):
    return np.maximum(0, Z)

# def sigmoid(a):
#     ans=1/(1+np.exp(-a))
#     return ans


def forward_prop(w1,b1,w2,b2,x):
    z1=np.dot(w1,x)+b1 #linear transformation
    a1=relu(z1) #non linear activation
    z2=np.dot(w2,a1)+b2 #second linear transformation
    exp_scores=np.exp(z2-np.max(z2,axis=0,keepdims=True)) #for numerical stability
    a2 = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)  # Softmax output
    return z1, a1, z2, a2

def cross_entropy_loss(a2,y):
    m=y.shape[1]
    loss=-np.sum(y*np.log(a2+1e-8))/m
    return loss

def backward_prop(w1,b1,w2,b2,z1,a1,z2,a2,x,y):
    m=x.shape[1]
    dz2=a2-y
    dw2=(1/m)*np.dot(dz2,a1.T)
    db2=(1/m)*np.sum(dz2,axis=1,keepdims=True)
    da1=np.dot(w2.T,dz2)
    dz1=da1*(z1>0)
    dw1=(1/m)*np.dot(dz1,x.T)
    db1=(1/m)*np.sum(dz1,axis=1,keepdims=True)
    return dw1,db1,dw2,db2

def update_params(w1,b1,w2,b2,dw1,db1,dw2,db2,lr):
    w1=w1-lr*dw1
    b1=b1-lr*db1
    w2=w2-lr*dw2
    b2=b2-lr*db2
    return w1,b1,w2,b2

def train(x,y,layers,epochs=10,lr=0.1,batch_size=256):
    w1,b1,w2,b2=layers
    n=x.shape[1]
    for epoch in range(epochs+1):
        # Shuffle data at the start of each epoch
        perm = np.random.permutation(n)
        x_shuffled = x[:, perm]
        y_shuffled = y[:, perm]
        for i in range(0, n, batch_size):
            X_batch = x_shuffled[:, i:i+batch_size]
            Y_batch = y_shuffled[:, i:i+batch_size]
            z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, X_batch)
            dw1, db1, dw2, db2 = backward_prop(w1, b1, w2, b2, z1, a1, z2, a2, X_batch, Y_batch)
            w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, lr)

        # Compute loss on full data at end of epoch
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, x)
        loss = cross_entropy_loss(a2, y)
        if epoch % max(1, (epochs//10)) == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return w1,b1,w2,b2

def accuracy(w1,b1,w2,b2,x,y):
    _,_,_,a2=forward_prop(w1,b1,w2,b2,x)
    predictions=np.argmax(a2,axis=0)
    labels=np.argmax(y,axis=0)
    acc=np.mean(predictions==labels)
    return acc

def run1(save_file="mnist_weights1.npz"):
    X_train = x_train.T  
    Y_train = y_train_enc 
    X_test = x_test.T    
    Y_test = y_test_enc
    w1, b1, w2, b2 = init_params()
    w1, b1, w2, b2 = train(X_train, Y_train, (w1, b1, w2, b2), 100, lr=0.1)
    train_acc = accuracy(w1, b1, w2, b2, X_train, Y_train)
    test_acc = accuracy(w1, b1, w2, b2, X_test, Y_test)
    print(f"Train Accuracy: {train_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")

    # Save weights and accuracies
    np.savez(save_file, w1=w1, b1=b1, w2=w2, b2=b2,
             train_acc=train_acc, test_acc=test_acc)
    
    print(f"Weights and accuracies saved to {save_file}")
    return w1, b1, w2, b2
    

run1()





