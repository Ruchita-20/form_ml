import React, { useState } from 'react';
import PredictorForm from './components/PredictorForm';

const App = () => {
  const [predictionResult, setPredictionResult] = useState(null);

  const handlePrediction = (result) => {
    setPredictionResult(result);
  };

  return (
    <div>
      <PredictorForm/>
    </div>
  );
};

export default App;
