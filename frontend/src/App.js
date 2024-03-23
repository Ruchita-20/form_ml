import React, { useState } from 'react';
import './App.css'; 
import PredictorForm from './components/PredictorForm';

const App = () => {
  const [showPopup, setShowPopup] = useState(false);

  const handleStartButtonClick = () => {
    setShowPopup(true);
  };

  const handleClosePopup = () => {
    setShowPopup(false);
  };

  return (
    <div className="App">
      <button onClick={handleStartButtonClick}>Start</button>
      {showPopup && <PredictorForm  csvfile='questions' onClose={handleClosePopup} />}
    </div>
  );
};

export default App;
