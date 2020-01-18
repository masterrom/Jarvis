# SmartDen (Blockchain Based)


# Purpose:
    The purpose of this project is to build a scalable platform that would use existing hardware (provided by user) and Google Machine Learning APIs (such Google Vision, Google Voice, AutoML) to provide advanced applications, such as capturing Business meetings, or a food recipe that you used to cook a dish at home or essentially any idea that one can think of.

    One amazing application exists in the security Industry. We know that most companies have existing cameras security 

    The purpose of this project is to essentially develop a smart webcam that uses google cloud vision api, and video labeler to analyze its view. The Result of the analysis will determine which the next possible action (ie, fire hazards, flooding, Breakin’s). Once that has been accomplished, additional feature can also be added to analyze the workspace which will be basically keep track of the people who are entering the premises.  Each entry of the person will then be recorded onto a blockchain network, which the upper management and the security is connected to.

## Phase of Developement
    Using the google cloud API’s, build a detection from hazards such as fire, flooding,... (which ever is easy)
    Build a basic UI using react, firebase, firestore
    Connect it with that
    Work towards parametrizing the field of vision ( ie be able select a specific area of vision)
    Work towards, who is walking in and out of the area, create a log of that
    Then try to implement facial recognition such that if someone who is not in the recognized list enters the premises, security is notified
    Record everything on a private blockchain network

## IBM watson API Usage
    Facial Recognition
    It can also be used to train to look for a specific area under its vision
    Might be possible to do this with google vision API
    Alternative solution if the above two are not immediately possible
    Crop out the vision, such that when the api is being used, only that specific area will be used when conducting analysis

## Google Vision Tasks
    Detect Environment Status
    Fire
    Smoke
    Detecting who exists within parameter
    Known profiles
    Use facial detection, to determine if the person is known
    Unknown profiles
    Use profile,a
    Tracking objects (ie, be able to tell the supervisor to keep track of a specific object)
    Keeps tracks of till it can, then gives last seen location of the object


### Getting Started


You need to authenticate yourself to the firebase database so follow the link at: https://firebase.google.com/docs/admin/setup/


### Python imports you need to install (pip)
## Firebase
> pip3 install firebase_admin
## Twilio
> pip3 install twilio
#### Set the environment variable for sid from your twillio account (mac)
> export TWILIO_ACCOUNT_SID=GD8ef67043**************1942g5c267
> export TWILIO_AUTH_TOKEN=435***********************54325

## Google Cloud Setup
- pip install -m google-cloud (Google-Cloud model works with python2)
Follow the below links for setup
- https://cloud.google.com/vision/docs/quickstart-client-libraries
- https://cloud.google.com/docs/authentication/getting-started

