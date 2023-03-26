#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install tensorflow


# In[3]:


pip install keras


# In[4]:


import tensorflow as tf
mnist = tf.keras.datasets.fashion_mnist


# In[6]:


(training_images,training_labels),(test_images,test_labels) = mnist.load_data()


# In[8]:


training_images = training_images.reshape(60000,28,28,1)
training_images = training_images/255.0
test_images = test_images.reshape(10000,28,28,1)
test_images = test_images/255.0


# In[9]:


from matplotlib import pyplot as plt
for i in range(10,20):
    plt.imshow(training_images[i][:,:,0], cmap =plt.cm.binary)
    plt.show()


# In[12]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(training_images,training_labels,epochs=7)
test_loss = model.evaluate(test_images,test_labels)


# In[18]:


predictions = model.predict(test_images[:5])
predictions


# In[19]:


import numpy as np
print(np.argmax(predictions,axis=1))
print(test_labels[:5])


# In[16]:


for i in range(0,5):
    image = test_images[i]
    image = np.array(image,dtype='float')
    pixels = image.reshape((28,28))
    plt.imshow(pixels, cmap='gray')
    plt.show()
    
    


# In[ ]:




