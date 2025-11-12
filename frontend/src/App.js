import React, { useState } from 'react';
import './App.css';
import ImageUpload from './components/ImageUpload';
import ResultDisplay from './components/ResultDisplay';
import Header from './components/Header';

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePrediction = async (formData) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Prediction failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'An error occurred during prediction');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <Header />
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <div className="bg-white rounded-lg shadow-xl p-6 md:p-8">
            <h2 className="text-2xl font-bold text-gray-800 mb-6 text-center">
              Diabetic Retinopathy Detection
            </h2>
            <p className="text-gray-600 mb-8 text-center">
              Upload a retinal scan image to detect diabetic retinopathy severity
            </p>

            {!result && (
              <ImageUpload
                onPredict={handlePrediction}
                loading={loading}
                error={error}
              />
            )}

            {result && (
              <ResultDisplay
                result={result}
                onReset={handleReset}
              />
            )}

            {error && !result && (
              <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-red-800">{error}</p>
                <button
                  onClick={handleReset}
                  className="mt-2 text-red-600 hover:text-red-800 underline"
                >
                  Try again
                </button>
              </div>
            )}
          </div>

          <div className="mt-8 bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">
              About This Tool
            </h3>
            <p className="text-gray-600 mb-4">
              DR.A-EYE uses deep learning (ResNet-18) to analyze retinal images and detect 
              diabetic retinopathy severity levels. The model has been trained on thousands 
              of retinal scans and achieves high accuracy in classification.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
              <div className="p-4 bg-blue-50 rounded-lg">
                <h4 className="font-semibold text-blue-900 mb-2">Severity Levels</h4>
                <ul className="text-sm text-blue-800 space-y-1">
                  <li>• No DR: No diabetic retinopathy</li>
                  <li>• Mild: Early stage retinopathy</li>
                  <li>• Moderate: Moderate severity</li>
                  <li>• Severe: Advanced stage</li>
                  <li>• Proliferate DR: Most severe stage</li>
                </ul>
              </div>
              <div className="p-4 bg-green-50 rounded-lg">
                <h4 className="font-semibold text-green-900 mb-2">Important Note</h4>
                <p className="text-sm text-green-800">
                  This tool is for screening purposes only. Always consult with a qualified 
                  ophthalmologist for professional diagnosis and treatment recommendations.
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
