/* import React, { useState } from 'react';

const ModelEvaluation = () => {
    const [responses, setResponses] = useState(Array(10).fill('10'));


  const handleInputChange = (e, index) => {
    const newResponses = [...responses];
        newResponses[index] = value;
        setResponses(newResponses);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          responses: responses,
        }),
      });

      if (response.ok) {
        const result = await response.json();
        // Handle the result as needed
        console.log('Evaluation successful:', result);
      } else {
        // Handle error response
        console.error('Evaluation failed');
      }
    } catch (error) {
      console.error('Error during evaluation:', error);
    }
  };

  return (
    <div>
      <h1>Model Evaluation</h1>
      <form onSubmit={handleSubmit}>
        <label htmlFor="person_id">Person ID:</label>
        <input
          type="text"
          id="person_id"
          name="person_id"
          value={personId}
          onChange={(e) => handleInputChange(e, null)}
          required
        />
        <br />
        <label>Answers to 10 Questions:</label>
        {[...Array(10).keys()].map((i) => (
          <div key={i}>
            <label htmlFor={`q${i + 1}`}>Question {i + 1}:</label>
            <input
              type="number"
              id={`q${i + 1}`}
              name={`q${i + 1}`}
              value={answers[i]}
              onChange={(e) => handleInputChange(e, i)}
              required
            />
            <br />
          </div>
        ))}
        <br />
        <input type="submit" value="Evaluate" />
      </form>
    </div>
  );
};

export default ModelEvaluation;
 */