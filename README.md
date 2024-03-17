<b><h1><center><font size="6">Chest X-Ray (Pneumonia) Binary Classification: <br> - CNN with Transfer Learning <br> (Tensorflow 2.0 & Keras)</font></center></h1></b>

## <b>1. | Introduction</b> 👋
  * Problem Overview 👨‍💻 </br>
    * 👉 The <mark><b>goal of this notebook</mark></b> is to <mark><b>determine which samples are from patients with Pneumonia</mark></b>.
    * 👉 The objective: <mark><b>train a convolutional neural network (CNN) able to successfully classify the chest X-ray images whether the result is NORMAL or PNEMONIA</b></mark>.
    * 👉 Therefore, <mark><b>it is a binary classification</b></mark>.
  * Dataset Description 🤔 </br>
    * 👉 The <mark><b>Chest X-ray Images Dataset is taken from Kaggle Dataset</b></mark>, <a href="https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images">Chest X-ray Images</a>.
    * 👉 This Dataset <mark><b>provides train folder and test folder with inside each folder has both NORMAL folder and PNEUMONIA folder respectively.</b></mark>.
    * <mark><b>Chest X-ray images (anterior-posterior)</mark></b> were <mark><b>selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children's Medical Center, Guangzhou</mark></b>. All chest X-ray imaging was <mark><b>performed as part of patients' routine clinical care</mark></b>.
    * For <mark><b>the analysis of chest x-ray images</mark></b>, <mark><b>all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans</mark></b>.
    * The <mark><b>diagnoses for the images</mark></b> were then <mark><b>graded by two expert physicians before being cleared for training the AI system</mark></b>. In order <mark><b>to account for any grading errors</mark></b>, the <mark><b>evaluation set was also checked by a third expert</mark></b>.
  * Analysis Introduction 🔎 </br>
    * 👉 In our case, with using <a href="https://www.image-net.org/">ImageNet</a> dataset, with <mark><b>more than 14 million images have been hand-annotated by the project to indicate what objects are pictured</mark></b>, in <mark><b>at least one million of the images, bounding boxes are also provided</mark></b> and <mark><b>contains more than 20,000 categories consisting of several hundred images for 1000 classes</mark></b>. This means that we <mark><b>can pick any CNN trained using ImageNet to get a warm start at training our own model</b></mark>.
    * 👉 <a href="https://en.wikipedia.org/wiki/Residual_neural_network">ResNet50</a> is a somewhat old, but still very popular, CNN. Its <mark><b>popularity come from the fact that it was the CNN that introduced the residual concept in deep learning</mark></b>. It <mark><b>also won the</mark></b> <a href="https://towardsdatascience.com/review-resnet-winner-of-ilsvrc-2015-image-classification-localization-detection-e39402bfa5d8">ILSVRC 2015</a> <mark><b>image classification contest</mark></b>. Since it is a well-known and very solid CNN, we decided to use it for our transfer learning task.
    * 👉 As the <mark><b>original ResNet50V2 was trained on ImageNet</mark></b>, <mark><b>its last layer outputs 1000 probabilities for a tested image to belong to the 1000 different ImageNet classes</mark></b>. Therefore, we <mark><b>cannot directly use it in our binary classification problem with only chest X-ray NORMAL and PNEUMONIA as classes</mark></b>.
    * 👉 Here we try <mark><b>2 approaches from transfer learning and compare their performances</mark></b>:
      > - <mark><b>using a pretrained model with frozing the base ResNet50V2 weights of its fully connected layers</mark></b> as the <mark><b>base for feature extraction</mark></b> and then <mark><b>add new layers and train them without changing anything in the convolutional section of the network</mark></b>.
      > - <mark><b>Fine Tuning, unfreezing the last layers of the pretrained model</mark></b> and then <mark><b>add new layers and train them in the convolutional section of the network</mark></b>.
    * 👉 In this case, <mark><b>the convolutional section becomes just an image feature extractor</mark></b> and <mark><b>the actual job of classifying the features is performed by the newly added fully connected layers</mark></b>.
  * Methods 🧾 </br>
    * Load a ResNet50V2 model trained using the ImageNet dataset.
    * 👉 <mark><b>Preprocess Images with Keras Image Data Generator</b></mark>,
      > - can <b><mark>rescale the pixel values</b></mark>
      > - can <b><mark>apply random transformation techniques for data augmentation on the fly</b></mark>.
      > - <b><mark>define two different generators</b></mark>,
        >> - The <b><mark>train_datagen<b><mark> includes <b><mark>some transformations to augment the train set</b></mark>.
        >> - The <b><mark>val_datagen<b><mark> is used to <b><mark>simply rescale the validation and test sets</b></mark>.
      > - <b><mark>apply those generators on each dataset using the flow_from_dataframe method</b></mark>.
      > - Apart from the transformations defined in each generator, <b><mark>the images are also resized based on the target_size set.</b></mark>.
    * During training process, we have applied some techniques like
      - 1st approach: Transfer Learning ResNet50V2 with all Frozen Fully Connected Layers
        - using image size of (224, 224, 3), BATCH_SIZE=128 and EPOCH=50.
        - adding global average pooling 2D layers to reduces the spatial dimensions to 1x1 while retaining the depth.
        - adding 10% dropout on input activation layers
        - utilizing Adam optimizer
        - monitoring validation loss using EarlyStopping at patience=5
        - monitoring validation loss using ReduceLROnPlateau at patience=2
      - 2nd approach: Transfer Learning ResNet50V2 with Fine-Tuning Selected Fully Connected Layers
        - using image size of (224, 224, 3) and EPOCH=50.
        - <mark><b>reducing the BATCH_SIZE from 128 to 32.</mark></b>
        - <mark><b>adding batch normalization layers</mark></b> to standardize the input and normalize hidden units of each prior layer of activation layers of a neural network by adjusting and scaling the activations and help reduce problem of covariant shift.
        - adding global average pooling 2D layers to reduces the spatial dimensions to 1x1 while retaining the depth.
        - adding 10% dropout on input activation layers
        - <mark><b>applying learning scheduler on Adam optimizer</mark></b>
        - monitoring validation loss using <mark><b>EarlyStopping with changing patience 5 to 15</mark></b>.
        - monitoring validation loss using <mark><b>ReduceLROnPlateau with changing patience 2 to 5</mark></b>.
    * Lastly, we can use the trained ResNet52V2 to predict the class of the preprocessed image.


## <b>2. | Accuracy of Best Model</b> 🧪
* Transfer Learning ResNet50V2 with all Frozen Fully Connected Layers
  - Training Accuracy achieved: 93.95%
  - Validation Accuracy achieved: 95.12%
  - Test F1 Score: 94.0%
* Transfer Learning ResNet50V2 with Fine-Tuning Selected Fully Connected Layers
  - Training Accuracy achieved: 97.94%
  - Validation Accuracy achieved:96.65%
  - Test F1 Score: 97.6%

## <b>3. | Conclusiion </b> 📤
* In this study respectively,
* With this transferred ResNet50V2, we can perform tests using any images having 224x224 resolution.
* The fine-tuning approach had reached the best score.
* Both 1st approach and 2nd approach trained models were quite generalized as their the differences between train loss and validation loss are small.
* The 2nd approach obviously outperforms 1st approach with 2nd approach's f1_score=97.6% and 1st approach's f1 score=94.0% respectively.
  > This further proves although we can make use of Resnet50V2 transfer learning's weights for higher start, greater slope and greater asymptotes, still needed with fine-tuning layers to adapt domains' information that not seen by models trained on 'imagenet' to reach better performance.
* The recall was close to 100%.
* Even without expertise on the medical field, it's reasonable to assume that false negatives are more 'costly' than false positives in this case.
* Reaching such recall with a relatively small dataset for training as this one, while also reaching a pretty good recall, is a good indicative of the model's capabilities confirmed by the high ROC-AUC value and double confirmed by high AUC value under precision recall curve for this small and unbalanced dataset.
* Correct predictions on some test images samples.

## <b>5. | Reference</b> 🔗
<ul><b><u>Github Notebook 📚</u></b>
        <li> <a style="color: #3D5A80" href="https://www.kaggle.com/code/jonaspalucibarbosa/chest-x-ray-pneumonia-cnn-transfer-learning">Chest X-Ray (Pneumonia) - CNN & Transfer Learning by JONAS PALUCI BARBOSA</a></li>
</ul>
<ul><b><u>Online Articles 🌏</u></b>
      <li><a style="color: #3D5A80" href="https://en.wikipedia.org/wiki/Transfer_learning">Transfer Learning by WIKIPEDIA</a></li>
      <li><a style="color: #3D5A80" href="https://en.wikipedia.org/wiki/Residual_neural_network">Residual neural network by WIKIPEDIA</a></li>
      <li><a style="color: #3D5A80" href="https://towardsdatascience.com/review-resnet-winner-of-ilsvrc-2015-image-classification-localization-detection-e39402bfa5d8">Review: ResNet — Winner of ILSVRC 2015 (Image Classification, Localization, Detection) by SIK-HO TSANG</a></li>
      <li><a style="color: #3D5A80" href="https://vijayabhaskar96.medium.com/tutorial-on-keras-flow-from-dataframe-1fd4493d237c">Tutorial on Keras flow_from_dataframe by VIJAYABHASKAR J</a></li>
      <li><a style="color: #3D5A80" href="https://www.tensorflow.org/guide/keras/transfer_learning">Tensorflow Core: Transfer learning & fine-tuning by TENSORFLOW</a></li>
      <li><a style="color: #3D5A80" href="https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator">TensorFlow v2.15.0.post1: tf.keras.preprocessing.image.ImageDataGenerator by TENSORFLOW</a></li>
      <li><a style="color: #3D5A80" href="https://keras.io/api/applications/">Keras 3 API documentation: Keras Applications by KERAS</a></li>
      <li><a style="color: #3D5A80" href="https://keras.io/api/applications/resnet/#resnet50v2-function">Keras 3 API documentation: Keras Applications/ResNet and ResNet50V2 by KERAS</a></li>
</ul>
<ul><b><u>Online Learning Channel 🌏</u></b>
        <li><a style="color: #3D5A80" href="https://www.udemy.com/course/artificial-intelligence-in-python-/learn/lecture/26598012#overview">Master Artificial Intelligence 2022 : Build 6 AI Projects by Dataisgood Academy</a></li>   
</ul>
