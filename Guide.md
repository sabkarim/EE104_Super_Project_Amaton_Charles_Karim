# **User Guide**

# *Part 4.1*
- For section 4.1 our team was able to implement a machine learning model to recognize 10 different imported color images using Keras
- The images chosen came from the CIFAR-10 database
  - contains 60,000 32x32 different captures
- The first 3 tests completed included VGG\_1 Block , VGG\_2 Block, and VGG\_3 Block
- The time ranged from 15-35 minutes for each run
- Moreover, other tests were completed for VGG3 and the main models for predicition and evaluation ar shown below
- ALL .py & HTML files are located in the "Option\_4\_Num\_1&quot" folder for each test

1. Final\_Model\_Evaluation
2. Final\_Model\_Predict
3. Final\_Model
4. VGG\_3\_Dropout
5. VGG\_3\_Data\_Augmentation
6. VGG\_3\_Dropout\_Regularization\_Data\_Augmentation\_Batch\_Linearization
7. VGG\_3\_Dropout\_Regularization\_Data\_Augmentation
8. VGG\_3\_Dropout\_Regularization
9. VGG\_3\_Weight\_Decay

An example of the prediction method is shown below using &quot;Final\_Model\_Predict&quot; file.

```
Python 3.8.5 (default, Sep 3 2020, 21:29:08) [MSC v.1916 64 bit (AMD64)]

Type &quot;copyright&quot;, &quot;credits&quot; or &quot;license&quot; for more information.

IPython 7.19.0 -- An enhanced Interactive Python.

In [**1**]: runfile(&#39;C:/Amaton\_Charles\_Karim/Option\_4\_Num\_1/Final\_Model\_Predict.py&#39;, wdir=&#39;C:/Amaton\_Charles\_Karim/Option\_4\_Num\_1&#39;)

2020-11-22 15:11:52.880287: I tensorflow/stream\_executor/platform/default/dso\_loader.cc:48] Successfully opened dynamic library cudart64\_101.dll

2020-11-22 15:11:52.880287: I tensorflow/stream\_executor/platform/default/dso\_loader.cc:48] Successfully opened dynamic library cudart64\_101.dll

2020-11-22 15:11:54.857136: I tensorflow/stream\_executor/platform/default/dso\_loader.cc:48] Successfully opened dynamic library nvcuda.dll

2020-11-22 15:11:54.885523: I tensorflow/core/common\_runtime/gpu/gpu\_device.cc:1716] Found device 0 with properties:

pciBusID: 0000:01:00.0 name: GeForce GTX 550 Ti computeCapability: 2.1

coreClock: 1.8GHz coreCount: 4 deviceMemorySize: 1.00GiB deviceMemoryBandwidth: 91.73GiB/s

2020-11-22 15:11:54.888032: I tensorflow/stream\_executor/platform/default/dso\_loader.cc:48] Successfully opened dynamic library cudart64\_101.dll

2020-11-22 15:11:54.892194: I tensorflow/stream\_executor/platform/default/dso\_loader.cc:48] Successfully opened dynamic library cublas64\_10.dll

2020-11-22 15:11:54.896028: I tensorflow/stream\_executor/platform/default/dso\_loader.cc:48] Successfully opened dynamic library cufft64\_10.dll

2020-11-22 15:11:54.897520: I tensorflow/stream\_executor/platform/default/dso\_loader.cc:48] Successfully opened dynamic library curand64\_10.dll

2020-11-22 15:11:54.902062: I tensorflow/stream\_executor/platform/default/dso\_loader.cc:48] Successfully opened dynamic library cusolver64\_10.dll

2020-11-22 15:11:54.904563: I tensorflow/stream\_executor/platform/default/dso\_loader.cc:48] Successfully opened dynamic library cusparse64\_10.dll

2020-11-22 15:11:54.913509: I tensorflow/stream\_executor/platform/default/dso\_loader.cc:48] Successfully opened dynamic library cudnn64\_7.dll

2020-11-22 15:11:54.916688: I tensorflow/core/common\_runtime/gpu/gpu\_device.cc:1812] Ignoring visible gpu device (device: 0, name: GeForce GTX 550 Ti, pci bus id: 0000:01:00.0, compute capability: 2.1) with Cuda compute capability 2.1. The minimum required Cuda capability is 3.5.

2020-11-22 15:11:54.926202: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x2139e4d2950 initialized for platform Host (this does not guarantee that XLA will be used). Devices:

2020-11-22 15:11:54.926733: I tensorflow/compiler/xla/service/service.cc:176] StreamExecutor device (0): Host, Default Version

2020-11-22 15:11:54.927121: I tensorflow/core/common\_runtime/gpu/gpu\_device.cc:1257] Device interconnect StreamExecutor with strength 1 edge matrix:

2020-11-22 15:11:54.927477: I tensorflow/core/common\_runtime/gpu/gpu\_device.cc:1263]

WARNING:tensorflow:From C:\Amaton\_Charles\_Karim\Option\_4\_Num\_1\Final\_Model\_Predict.py:30: Sequential.predict\_classes (from tensorflow.python.keras.engine.sequential) is deprecated and will be removed after 2021-01-01.

Instructions for updating:

Please use instead:\* `np.argmax(model.predict(x), axis=-1)`, if your model does multi-class classification (e.g. if it uses a `softmax` last-layer activation).\* `(model.predict(x) \&gt; 0.5).astype(&quot;int32&quot;)`, if your model does binary classification (e.g. if it uses a `sigmoid` last-layer activation).

CIFAR-10 Classes (0-9):

[&#39;airplane&#39;, &#39;automobile&#39;, &#39;bird&#39;, &#39;cat&#39;, &#39;deer&#39;, &#39;dog&#39;, &#39;frog&#39;, &#39;horse&#39;, &#39;ship&#39;, &#39;truck&#39;]

Image Under CIFAR-10 Class:

4

Time Elapsed: 2.464463233947754
```
In [**2**]:
![processed deer](../Option_4_Num_2_3_4/Deer_1_database_processed.png)

![alt text](RackMultipart20201210-4-b5mg95_html_d8f458ef325a479e.png)

# *Part 4.2 & 4.3*
- For the following section our team was able to successfully resize 20 chosen images from each of the following categories: 
  - Airplane, Automobile, Bird, Cat, Dog, Deer , Horse, Frog, Ship, Frog, and Truck
- The pictures were found using google and 2 pictures were selected for each category
- The original 32x32 images were re-sized to a 32x32x3 pixel format and saved into the "Option\_4\_Num\_2\_3\_4" folder
  - In other words, the images were made smaller, while also maintaining color
- To run the coda a .py file is provided in the "Option\_4\_Num\_2\_3\_4" folder that is labeled "Option\_4\_Num\_2.py"
- As shown below our team re-sized a .png image of a horse. The left image is the original and the right is the re-sized version.

![alt text](RackMultipart20201210-4-b5mg95_html_6407f926b49e25f6.png) ![alt text](RackMultipart20201210-4-b5mg95_html_1e2c846ff3b3b342.png)

# *Part 4.4*
- For the last section our team then proceed to use the 23x23x3 images from the previous task to check the accuracy of the original program created in section 1
  - The code was created to first import all the images into the database then procced to ask the user, which image it would like to select
- After the user selected an image the program would predict the category of the image and the probability of the predicted category being true
- It is important to note that side profiles images and pictures of cats, dogs, &amp; deers usually resulted in wrong predictions
- To obtain the code navigate to the "Option\_4\_Num\_2\_3\_4" folder and selected the file labeled "Option\_4\_Num\_3\_4.py"
- Furthermore, as shown below our program was able to successfully run with a 100% prediction

![](RackMultipart20201210-4-b5mg95_html_f48e51f8b0724439.png)
