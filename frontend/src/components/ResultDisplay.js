import React from 'react';

function ResultDisplay({ result, onReset }) {
  const getSeverityConfig = (severity) => {
    const configs = {
      'No_DR': {
        color: 'from-green-400 to-emerald-600',
        bg: 'bg-gradient-to-br from-green-50 to-emerald-50',
        text: 'text-green-800',
        border: 'border-green-400',
        icon: '✅',
        emoji: '🎉',
        label: 'No Diabetic Retinopathy',
        message: 'Excellent news!'
      },
      'Mild': {
        color: 'from-yellow-400 to-amber-500',
        bg: 'bg-gradient-to-br from-yellow-50 to-amber-50',
        text: 'text-yellow-800',
        border: 'border-yellow-400',
        icon: '⚠️',
        emoji: '👁️',
        label: 'Mild Diabetic Retinopathy',
        message: 'Early detection is key!'
      },
      'Moderate': {
        color: 'from-orange-400 to-red-500',
        bg: 'bg-gradient-to-br from-orange-50 to-red-50',
        text: 'text-orange-800',
        border: 'border-orange-400',
        icon: '⚠️',
        emoji: '🔍',
        label: 'Moderate Diabetic Retinopathy',
        message: 'Action recommended'
      },
      'Severe': {
        color: 'from-red-500 to-red-700',
        bg: 'bg-gradient-to-br from-red-50 to-pink-50',
        text: 'text-red-800',
        border: 'border-red-500',
        icon: '🚨',
        emoji: '⚡',
        label: 'Severe Diabetic Retinopathy',
        message: 'Immediate attention needed'
      },
      'Proliferate_DR': {
        color: 'from-red-700 to-red-900',
        bg: 'bg-gradient-to-br from-red-100 to-pink-100',
        text: 'text-red-900',
        border: 'border-red-600',
        icon: '🚨',
        emoji: '⚡',
        label: 'Proliferative Diabetic Retinopathy',
        message: 'Urgent care required'
      }
    };
    return configs[severity] || configs['No_DR'];
  };

  const config = getSeverityConfig(result.predicted_class);

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Main Result Card */}
      <div className={`relative overflow-hidden rounded-3xl border-4 ${config.border} ${config.bg} shadow-2xl transform hover:scale-[1.01] transition-transform`}>
        <div className={`absolute top-0 right-0 w-96 h-96 bg-gradient-to-br ${config.color} opacity-20 rounded-full -mr-48 -mt-48 blur-3xl`}></div>
        <div className="relative p-10">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-8">
            <div className="flex items-center space-x-5 mb-6 md:mb-0">
              <div className={`w-24 h-24 bg-gradient-to-br ${config.color} rounded-3xl flex items-center justify-center text-5xl shadow-2xl transform hover:rotate-12 transition-transform`}>
                {config.emoji}
              </div>
              <div>
                <div className="text-sm font-bold text-gray-600 mb-2 uppercase tracking-wide">{config.message}</div>
                <h3 className="text-4xl font-extrabold text-gray-900 mb-2">Prediction Result</h3>
                <p className={`text-2xl font-bold ${config.text}`}>{config.label}</p>
              </div>
            </div>
            <div className="text-center md:text-right">
              <div className="inline-block bg-white rounded-2xl px-8 py-6 shadow-xl border-2 border-gray-100">
                <div className="text-5xl font-extrabold bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 bg-clip-text text-transparent mb-2">
                  {result.confidence_percentage}%
                </div>
                <div className="text-base font-bold text-gray-600">Confidence Score</div>
                <div className="mt-2 w-32 h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div 
                    className={`h-full bg-gradient-to-r ${config.color} rounded-full transition-all duration-1000`}
                    style={{ width: `${result.confidence_percentage}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white/90 backdrop-blur-sm rounded-2xl p-7 border-2 border-white/50 shadow-lg">
              <div className="flex items-start">
                <div className="w-12 h-12 bg-gradient-to-br from-blue-400 to-indigo-500 rounded-xl flex items-center justify-center mr-4 flex-shrink-0">
                  <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <div>
                  <p className="font-bold text-gray-900 mb-3 text-lg">Analysis Message</p>
                  <p className="text-gray-700 leading-relaxed text-base">{result.message}</p>
                </div>
              </div>
            </div>

            <div className="bg-white/90 backdrop-blur-sm rounded-2xl p-7 border-2 border-white/50 shadow-lg">
              <div className="flex items-start">
                <div className="w-12 h-12 bg-gradient-to-br from-indigo-400 to-purple-500 rounded-xl flex items-center justify-center mr-4 flex-shrink-0">
                  <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                  </svg>
                </div>
                <div>
                  <p className="font-bold text-gray-900 mb-3 text-lg">Recommendation</p>
                  <p className="text-gray-700 leading-relaxed text-base">{result.recommendation}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Probability Breakdown */}
      <div className="bg-gradient-to-br from-white to-blue-50/30 rounded-3xl shadow-2xl p-10 border-2 border-gray-100">
        <div className="flex items-center mb-8">
          <div className="w-16 h-16 bg-gradient-to-br from-blue-500 via-indigo-600 to-purple-600 rounded-2xl flex items-center justify-center mr-5 shadow-xl">
            <svg className="w-9 h-9 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h4 className="text-3xl font-extrabold text-gray-900">Detailed Probability Breakdown</h4>
        </div>
        <div className="space-y-5">
          {Object.entries(result.all_probabilities)
            .sort((a, b) => b[1] - a[1])
            .map(([className, probability]) => {
              const isPredicted = className === result.predicted_class;
              const probConfig = getSeverityConfig(className);
              return (
                <div key={className} className="space-y-3 bg-white/70 backdrop-blur-sm rounded-xl p-5 border-2 border-gray-100 hover:border-blue-300 transition-all">
                  <div className="flex justify-between items-center">
                    <div className="flex items-center space-x-3">
                      <span className="text-2xl">{probConfig.emoji}</span>
                      <span className={`text-lg font-bold ${isPredicted ? 'text-gray-900' : 'text-gray-700'}`}>
                        {probConfig.label}
                      </span>
                      {isPredicted && (
                        <span className="px-3 py-1 bg-gradient-to-r from-blue-500 to-indigo-600 text-white text-xs font-bold rounded-full shadow-md animate-pulse">
                          PREDICTED
                        </span>
                      )}
                    </div>
                    <span className={`text-2xl font-extrabold ${isPredicted ? 'text-blue-600' : 'text-gray-600'}`}>
                      {(probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden shadow-inner">
                    <div
                      className={`h-4 rounded-full transition-all duration-1000 ease-out ${
                        isPredicted
                          ? `bg-gradient-to-r ${probConfig.color} shadow-lg`
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
          className="inline-flex items-center px-10 py-5 bg-gradient-to-r from-gray-700 via-gray-800 to-gray-900 text-white rounded-2xl font-bold text-lg shadow-2xl hover:shadow-3xl hover:from-gray-800 hover:via-gray-900 hover:to-black transform hover:scale-105 transition-all duration-200"
        >
          <svg className="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Analyze Another Image
        </button>
      </div>
    </div>
  );
}

export default ResultDisplay;
