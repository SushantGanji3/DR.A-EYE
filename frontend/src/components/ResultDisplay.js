import React from 'react';

function ResultDisplay({ result, onReset }) {
  const getSeverityColor = (severity) => {
    const colors = {
      'No_DR': 'bg-green-100 text-green-800 border-green-300',
      'Mild': 'bg-yellow-100 text-yellow-800 border-yellow-300',
      'Moderate': 'bg-orange-100 text-orange-800 border-orange-300',
      'Severe': 'bg-red-100 text-red-800 border-red-300',
      'Proliferate_DR': 'bg-red-200 text-red-900 border-red-400',
    };
    return colors[severity] || 'bg-gray-100 text-gray-800 border-gray-300';
  };

  const getSeverityIcon = (severity) => {
    if (severity === 'No_DR') {
      return '✓';
    } else if (severity === 'Proliferate_DR') {
      return '⚠';
    } else {
      return '!';
    }
  };

  return (
    <div className="space-y-6">
      <div className={`p-6 rounded-lg border-2 ${getSeverityColor(result.predicted_class)}`}>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className="text-3xl font-bold">
              {getSeverityIcon(result.predicted_class)}
            </div>
            <div>
              <h3 className="text-2xl font-bold">Prediction Result</h3>
              <p className="text-lg opacity-80">{result.predicted_class}</p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold">
              {result.confidence_percentage}%
            </div>
            <div className="text-sm opacity-80">Confidence</div>
          </div>
        </div>

        <div className="mt-4 p-4 bg-white bg-opacity-50 rounded-lg">
          <p className="font-semibold mb-2">Message:</p>
          <p>{result.message}</p>
        </div>

        <div className="mt-4 p-4 bg-white bg-opacity-50 rounded-lg">
          <p className="font-semibold mb-2">Recommendation:</p>
          <p>{result.recommendation}</p>
        </div>
      </div>

      <div className="bg-gray-50 rounded-lg p-6">
        <h4 className="font-semibold text-gray-800 mb-4">All Class Probabilities</h4>
        <div className="space-y-3">
          {Object.entries(result.all_probabilities)
            .sort((a, b) => b[1] - a[1])
            .map(([className, probability]) => (
              <div key={className} className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="font-medium text-gray-700">{className}</span>
                  <span className="text-gray-600">{(probability * 100).toFixed(2)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${
                      className === result.predicted_class
                        ? 'bg-blue-600'
                        : 'bg-gray-400'
                    }`}
                    style={{ width: `${probability * 100}%` }}
                  ></div>
                </div>
              </div>
            ))}
        </div>
      </div>

      <div className="flex justify-center">
        <button
          onClick={onReset}
          className="px-6 py-3 bg-gray-600 text-white rounded-lg font-semibold hover:bg-gray-700 transition-colors"
        >
          Analyze Another Image
        </button>
      </div>
    </div>
  );
}

export default ResultDisplay;

