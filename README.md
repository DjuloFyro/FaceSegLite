<!-- PROJECT LOGO -->
<br />
<div align="center">
  <!--<a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>-->

  <h1 align="center">FaceSegLite</h1>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
      </ul>
    </li>
    <li>
      <a href="#train-model">Train Model</a>
      <ul>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributors">Contributors</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
This project's objective is to develop a model that excels in instance-based segmentation with high efficiency. We will specifically focus on segmenting faces to ensure the project's manageability within our available resources and timeframe.

When we refer to performance, we are targeting a model that achieves high accuracy in face segmentation, yet remains compact enough for real-time operation.

Hence, the primary aim is to create a model that can seamlessly function with a webcam through an easy-to-use API, achieving an optimal compromise between segmentation quality and the model's compactness.

<!-- Train Model -->
## Train Model
To train the model from scratch you first need to load our dataset from HuggingFace.
From the root directory:
  ```sh
git clone https://huggingface.co/datasets/Djulo/Wider_FaceSegLite data/
  ```

Then you can follow the instructions in the notebook `notebooks/fsl_building_model.ipynb` to train the model.

There are also other notebooks available in the `notebooks` directory to train a model with different architectures and hyperparameters.

You can also use the pre-trained models available in the `models` directory.

<!-- USAGE EXAMPLES -->
## Usage
To use the model, you can either use our pre-trained models or train your own model.

To test our project, you can do the following:
  ```sh
  git clone https://github.com/DjuloFyro/FaceSegLite.git
  ```
  ```sh
  cd FaceSegLite
  ```
Create a virtual environment and install the requirements:
  ```sh
  python3 -m venv venv
  ```
  ```sh
  source venv/bin/activate
  ```
  ```sh
  pip install -r requirements.txt
  ```
Run the following command to start the backend server:
  ```sh
  cd frontend
  ```
  ```sh
  flask --app backend.py run
  ```
Open a new terminal, go to the `frontend` folder and run the following command to start the frontend server:

  ```sh
  streamlit run frontend.py
  ```
Open your browser and go to the following address:
  ```sh
  http://localhost:8501
  ```
You can now use the application to test our different models.


<!-- CONTRIBUTORS -->
## Contributors
- Julian Gil
- Antoine Feret
- Habib Adoum Mandazou
- Theo Tinti
  
<!-- LICENSE -->
## License
Distributed under the MIT License. See `LICENSE.txt` for more information.

<!-- CONTACT -->
## Contact
Julian Gil - juliangil2424@gmail.com

