import React from 'react';

function ResultDisplay({ result, onReset }) {
  const getSeverityConfig = (severity) => {
    const configs = {
      'No_DR': {
        color: 'from-green-500 to-emerald-600',
        bg: 'bg-green-50',
        text: 'text-green-800',
        border: 'border-green-300',
        icon: (
          <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        ),
        label: 'No Diabetic Retinopathy'
      },
      'Mild': {
        color: 'from-yellow-400 to-amber-500',
        bg: 'bg-yellow-50',
        text: 'text-yellow-800',
        border: 'border-yellow-300',
        icon: (
          <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        ),
        label: 'Mild Diabetic Retinopathy'
      },
      'Moderate': {
        color: 'from-orange-500 to-red-500',
        bg: 'bg-orange-50',
        text: 'text-orange-800',
        border: 'border-orange-300',
        icon: (
          <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        ),
        label: 'Moderate Diabetic Retinopathy'
      },
      'Severe': {
        color: 'from-red-500 to-red-700',
        bg: 'bg-red-50',
        text: 'text-red-800',
        border: 'border-red-300',
        icon: (
          <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        ),
        label: 'Severe Diabetic Retinopathy'
      },
      'Proliferate_DR': {
        color: 'from-red-700 to-red-900',
        bg: 'bg-red-100',
        text: 'text-red-900',
        border: 'border-red-400',
        icon: (
          <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        ),
        label: 'Proliferative Diabetic Retinopathy'
      }
    };
    return configs[severity] || configs['No_DR'];
  };

  const config = getSeverityConfig(result.predicted_class);

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Main Result Card */}
      <div className={`relative overflow-hidden rounded-2xl border-2 ${config.border} ${config.bg} shadow-xl`}>
        <div className={`absolute top-0 right-0 w-64 h-64 bg-gradient-to-br ${config.color} opacity-10 rounded-full -mr-32 -mt-32`}></div>
        <div className="relative p-8">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6">
            <div className="flex items-center space-x-4 mb-4 md:mb-0">
              <div className={`w-16 h-16 bg-gradient-to-br ${config.color} rounded-xl flex items-center justify-center text-white shadow-lg`}>
                {config.icon}
              </div>
              <div>
                <h3 className="text-3xl font-bold text-gray-900 mb-1">Prediction Result</h3>
                <p className={`text-xl font-semibold ${config.text}`}>{config.label}</p>
              </div>
            </div>
            <div className="text-center md:text-right">
              <div className="inline-block bg-white rounded-xl px-6 py-4 shadow-lg">
                <div className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                  {result.confidence_percentage}%
                </div>
                <div className="text-sm font-medium text-gray-600 mt-1">Confidence</div>
              </div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white/70 backdrop-blur-sm rounded-xl p-6 border border-white/50">
              <div className="flex items-start">
                <svg className="w-6 h-6 text-blue-600 mr-3 mt-1 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div>
                  <p className="font-semibold text-gray-900 mb-2">Analysis Message</p>
                  <p className="text-gray-700 leading-relaxed">{result.message}</p>
                </div>
              </div>
            </div>

            <div className="bg-white/70 backdrop-blur-sm rounded-xl p-6 border border-white/50">
              <div className="flex items-start">
                <svg className="w-6 h-6 text-indigo-600 mr-3 mt-1 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
                <div>
                  <p className="font-semibold text-gray-900 mb-2">Recommendation</p>
                  <p className="text-gray-700 leading-relaxed">{result.recommendation}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Probability Breakdown */}
      <div className="bg-white rounded-2xl shadow-lg p-8 border border-gray-100">
        <div className="flex items-center mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg flex items-center justify-center mr-4">
            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h4 className="text-2xl font-bold text-gray-900">Probability Distribution</h4>
        </div>
        <div className="space-y-4">
          {Object.entries(result.all_probabilities)
            .sort((a, b) => b[1] - a[1])
            .map(([className, probability]) => {
              const isPredicted = className === result.predicted_class;
              const probConfig = getSeverityConfig(className);
              return (
                <div key={className} className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className={`font-semibold ${isPredicted ? 'text-gray-900' : 'text-gray-700'}`}>
                      {probConfig.label}
                      {isPredicted && (
                        <span className="ml-2 px-2 py-0.5 bg-blue-100 text-blue-700 text-xs font-bold rounded-full">
                          PREDICTED
                        </span>
                      )}
                    </span>
                    <span className={`text-lg font-bold ${isPredicted ? 'text-blue-600' : 'text-gray-600'}`}>
                      {(probability * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                    <div
                      className={`h-3 rounded-full transition-all duration-1000 ease-out ${
                        isPredicted
                          ? `bg-gradient-to-r ${probConfig.color}`
                          : 'bg-gray-400'
                      }`}
                      style={{ width: `${probability * 100}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
        </div>
      </div>

      {/* Action Button */}
      <div className="flex justify-center">
        <button
          onClick={onReset}
          className="inline-flex items-center px-8 py-4 bg-gradient-to-r from-gray-700 to-gray-800 text-white rounded-xl font-semibold text-lg shadow-lg hover:shadow-xl hover:from-gray-800 hover:to-gray-900 transform hover:scale-105 transition-all duration-200"
        >
          <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Analyze Another Image
        </button>
      </div>
    </div>
  );
}

export default ResultDisplay;
