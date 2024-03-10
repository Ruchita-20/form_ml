import React, { useState, useEffect } from 'react';
import axios from 'axios';

const PredictorForm = ({ handlePrediction }) => {
  const [responses, setResponses] = useState(Array(10).fill('10'));
  const [result, setResult] = useState(null);

  useEffect(() => {
    if (result !== null) {
      console.log('Result State:', result);
    }
  }, [result]);

  const handleResponseChange = (index, value) => {
    const newResponses = [...responses];
    newResponses[index] = value;
    setResponses(newResponses);
  };

  const handleSubmit = async (e) => {
    e.preventDefault(); // Prevent the default form submission behavior

    try {
      const res = await axios.post('http://localhost:5000/predict', {
        col0: responses[0],
        col1: responses[1],
        col2: responses[2],
        col3: responses[3],
        col4: responses[4],
        col5: responses[5],
        col6: responses[6],
        col7: responses[7],
        col8: responses[8],
        col9: responses[9],
      });
      console.log('Response from server:', res.data);
      setResult({
        predicted_total: res.data.predicted_total,
      });
    } catch (error) {
      console.log('Error predicting result', error);
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        {responses.map((value, index) => (
          <div key={index}>
            <label htmlFor={`col${index}`}>Question {index + 1}:</label>
            <select
              name={`col${index}`}
              id={`col${index}`}
              value={value}
              onChange={(e) => handleResponseChange(index, e.target.value)}
            >
              <option value="10">10</option>
              <option value="20">20</option>
              <option value="30">30</option>
            </select>
            <br />
          </div>
        ))}
        <button type="submit">Predict</button>
      </form>
      {result !== null && (
        <div>
          <h2>Prediction Result:</h2>
          <p>Predicted Total: {result.predicted_total}</p>
        </div>
      )}
    </div>
  );
};

export default PredictorForm;
