# notHotdog

Real-Time Multi-Object Recognition via Smartphone

## Motivation
The motivation of this project is to aid tourists in a foreign country and language learners by providing easy and fast lookup of the name of previously unknown objects and thus improving one’s vocabulary size.  

## Components
- TensorFlow
- CoreML
- ARKit

## System Architecture
![System_Architecture](https://github.com/dliuproduction/notHotDog/blob/master/systemArchitecture.png)

## Requirements
- Appropriate dataset
- Ability to track object once detected
- Ability to detect multiple objects
- Ability to distinguish representative objects 
- Ability to display labels in multiple languages
## Constraints
- Hardware: limited processor frequency and memory availability of Smartphone
- Refresh rate: refresh rate needs to be high enough to track moving objects
- Detection environment: Object detection can only perform optimally in ideal lighting conditions.
- Background: Detecting specific objects in different contexts and over different backgrounds can be challenging

## Process
1. Built a single detection model architecture using Keras which runs on top of TensorFlow
2. Attempted to use Google Cloud Compute and training the model on the cloud from scratch
3. Decided to use transfer learning and fine tuning on pretrained weights
4. Determined the most suitable dataset based on priorities
5. Built app with different models
6. Added multi-language support
7. Incorporated MobileNet-SSD model with multi-object detection at 7fps with ARkit location tracking
## Applications
1. In a new environment or a new city, a language learner uses the app to learn basic vocabulary in the language they choose 
2. children learning new words use the app to explore language and the world around them on their own which has been shown to create a positive learning experience

## Results & Conclusion
We created an application capable of recognizing 91 common objects in real-time on a Smartphone using its camera view. The object names can also be translated to 10 different languages and displayed in an augmented reality (AR) view with scaled sizes and orientations.  
