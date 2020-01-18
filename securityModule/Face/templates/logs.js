
const firebase = require("firebase");
// Required for side-effects
require("firebase/firestore");


// Your web app's Firebase configuration
var firebaseConfig = {
    apiKey: "AIzaSyBgkMteibGnVsRCDlDdZXbBAU2zuSfkrdc",
    authDomain: "supervisor-f2f29.firebaseapp.com",
    databaseURL: "https://supervisor-f2f29.firebaseio.com",
    projectId: "supervisor-f2f29",
    storageBucket: "supervisor-f2f29.appspot.com",
    messagingSenderId: "124766684889",
    appId: "1:124766684889:web:f657ab26b2fbe51b4920c8"
  };
  // Initialize Firebase
  firebase.initializeApp(firebaseConfig);
  firebase.firestore().settings( { timestampsInSnapshots: true }) 


  var db = firebase.firestore();



var citiesRef = db.collection("cities");

citiesRef.doc("SF").set({
name: "San Francisco", state: "CA", country: "USA",
capital: false, population: 860000,
regions: ["west_coast", "norcal"] });

//   db.collection("Camera").doc("camera1")
//     .onSnapshot({
//         // Listen for document metadata changes
//         includeMetadataChanges: true
//     }, function(doc) {
    
//         console.log(" data: ", doc.data());
//     });