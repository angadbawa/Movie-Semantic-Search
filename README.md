# Movie Semantic Search 


## Introduction

Movie data has witnessed exponential growth on the Internet, necessitating the analysis of multimedia content using computer vision techniques. This article presents a novel three-fold framework based on intelligent Convolutional Neural Networks (CNN) for scene segmentation in movies. The framework integrates shot segmentation, object detection, and object-based shot matching to accurately detect scene boundaries. The first fold segments the input movie into shots, the second fold detects objects in the segmented shots and the third fold performs object-based shots matching for detecting scene boundaries. Texture and shape features are fused for shots segmentation, and each shot is represented by a set of detected objects acquired from a lightweight CNN model. Finally, we apply set theory with the sliding window–based approach to integrate the same shots to decide scene boundaries. The experimental evaluation indicates that our proposed approach outran the existing movie scene segmentation approaches. The proposed semantic-based navigation system has the potential to revolutionize the way users interact with movies. By offering an intuitive and contextually meaningful navigation experience, users can easily navigate to specific scenes or moments of interest without having to rely on conventional timestamp-based navigation methods.

We are Reproducing the paper titled "Movie scene segmentation using object detection and set theory " by "Ijaz Ul Haq1 , Khan Muhammad2, Tanveer Hussain1, Soonil Kwon2, Maleerat Sodanil3, Sung Wook Baik1 and Mi Young Lee1 "

## Installation
```
Clone the repository : <code>git clone https://github.com/angadbawa/Movie-Semantic-Search </code>

Video Caption Generator: <code>cd Movie-Semantic-Search </code>

Create environment: <code>conda create -n MSS python=3.7</code>

Activate environment: <code>conda activate MSS</code>

Install requirements: <code>pip install -r requirements.txt</code>
```

## Usage
```
1. Clone the repository: git clone https://github.com/angadbawa/Movie-Semantic-Search.git
2. Open the Shot_segmentation.v2.py.
3. Run the code cells to generate the all the shots from the input movie.(Note it may take a while depending on your GPU Speed)
4. The output shots will be saved in .mp4 format.
5. Open the Movie Semantic Search.ipynb Jupyter Notebook.
6. Run the remaining code generate the compiled scenes.
7. The output scenes will be saved in .mp4 format.

```

## Input
```
The model is reproduced on the movie " Devil Wears Prada " and that is taken as input
```

## Output
``` 
The output shots and scenes will be saved in .mp4 format. The file will 
be located in the 'Movie Semantic Search'  directory.
```

## Sample Scenes Generated


https://github.com/angadbawa/Movie-Semantic-Search/assets/86665541/23120814-be03-4ced-9ea1-998e808cb371

https://github.com/angadbawa/Movie-Semantic-Search/assets/86665541/984a6184-f755-4be8-ad54-6baee9f1d770





