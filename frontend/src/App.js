import React, { useState } from 'react';
import './App.css';
import ImageUpload from './components/ImageUpload';
import ResultDisplay from './components/ResultDisplay';
import Header from './components/Header';
import Footer from './components/Footer';

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
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 flex flex-col">
      <Header />
      <main className="flex-grow container mx-auto px-4 py-12 max-w-6xl">
        {/* Hero Section */}
        {!result && (
          <div className="text-center mb-12 animate-fade-in">
            <div className="inline-flex items-center justify-center w-24 h-24 bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-500 rounded-3xl shadow-2xl mb-8 transform hover:scale-110 transition-transform duration-300">
              <svg className="w-14 h-14 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
            </div>
            <h1 className="text-6xl font-extrabold text-gray-900 mb-6 bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 bg-clip-text text-transparent leading-tight">
              Welcome to DR.A-EYE
            </h1>
            <p className="text-2xl text-gray-700 max-w-3xl mx-auto leading-relaxed mb-4 font-medium">
              Your trusted AI companion for early diabetic retinopathy detection
            </p>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Simply upload a retinal scan image and get instant, accurate analysis powered by advanced deep learning technology
            </p>
            
            {/* Trust Indicators */}
            <div className="flex flex-wrap items-center justify-center gap-8 mt-10">
              <div className="flex items-center space-x-2 bg-white/80 backdrop-blur-sm px-4 py-2 rounded-full shadow-md">
                <svg className="w-5 h-5 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                <span className="text-sm font-semibold text-gray-700">82% Accuracy</span>
              </div>
              <div className="flex items-center space-x-2 bg-white/80 backdrop-blur-sm px-4 py-2 rounded-full shadow-md">
                <svg className="w-5 h-5 text-blue-500" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M10.394 2.08a1 1 0 00-.788 0l-7 3a1 1 0 000 1.84L5.25 8.051a.999.999 0 01.356-.257l4-1.714a1 1 0 11.788 1.838L7.667 9.088l1.94.831a1 1 0 00.787 0l7-3a1 1 0 000-1.838l-7-3zM3.31 9.397L5 10.12v4.102a8.969 8.969 0 00-1.05-.174 1 1 0 01-.89-.89 11.115 11.115 0 01.25-3.762zM9.3 16.573A9.026 9.026 0 007 14.935v-3.957l1.818.78a3 3 0 002.364 0l5.508-2.361a11.026 11.026 0 01.25 3.762 1 1 0 01-.89.89 8.968 8.968 0 00-5.35 2.524 1 1 0 01-1.4 0zM6 18a1 1 0 001-1v-2.065a8.935 8.935 0 00-2-.712V17a1 1 0 001 1z" />
                </svg>
                <span className="text-sm font-semibold text-gray-700">AI-Powered</span>
              </div>
              <div className="flex items-center space-x-2 bg-white/80 backdrop-blur-sm px-4 py-2 rounded-full shadow-md">
                <svg className="w-5 h-5 text-purple-500" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z" clipRule="evenodd" />
                </svg>
                <span className="text-sm font-semibold text-gray-700">Instant Results</span>
              </div>
            </div>
          </div>
        )}

        {/* Main Content Card */}
        <div className="bg-white/90 backdrop-blur-lg rounded-3xl shadow-2xl overflow-hidden border border-white/50 transform hover:shadow-3xl transition-shadow duration-300">
          <div className="p-8 md:p-12">
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
              <div className="mt-6 p-6 bg-gradient-to-r from-red-50 to-pink-50 border-l-4 border-red-500 rounded-xl animate-shake shadow-lg">
                <div className="flex items-start">
                  <div className="flex-shrink-0">
                    <svg className="w-8 h-8 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <div className="ml-4 flex-1">
                    <h3 className="text-lg font-bold text-red-800 mb-2">Oops! Something went wrong</h3>
                    <p className="text-red-700 mb-4">{error}</p>
                    <button
                      onClick={handleReset}
                      className="px-6 py-2 bg-gradient-to-r from-red-500 to-pink-500 text-white rounded-lg font-semibold hover:from-red-600 hover:to-pink-600 transition-all transform hover:scale-105 shadow-md"
                    >
                      Try Again
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Information Section */}
        {!result && (
          <div className="mt-12 grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-2xl shadow-xl p-8 border border-blue-100 hover:shadow-2xl transition-all transform hover:-translate-y-1">
              <div className="flex items-center mb-6">
                <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl flex items-center justify-center mr-4 shadow-lg">
                  <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <h3 className="text-2xl font-bold text-gray-900">Understanding Results</h3>
              </div>
              <ul className="space-y-4 text-gray-700">
                <li className="flex items-start group">
                  <span className="w-3 h-3 bg-green-500 rounded-full mr-3 mt-2 group-hover:scale-150 transition-transform"></span>
                  <div>
                    <span className="font-semibold text-gray-900">No DR:</span> 
                    <span className="ml-2">Great news! No diabetic retinopathy detected. Continue regular checkups.</span>
                  </div>
                </li>
                <li className="flex items-start group">
                  <span className="w-3 h-3 bg-yellow-500 rounded-full mr-3 mt-2 group-hover:scale-150 transition-transform"></span>
                  <div>
                    <span className="font-semibold text-gray-900">Mild:</span> 
                    <span className="ml-2">Early stage detected. Consult your ophthalmologist for monitoring.</span>
                  </div>
                </li>
                <li className="flex items-start group">
                  <span className="w-3 h-3 bg-orange-500 rounded-full mr-3 mt-2 group-hover:scale-150 transition-transform"></span>
                  <div>
                    <span className="font-semibold text-gray-900">Moderate:</span> 
                    <span className="ml-2">Moderate severity. Schedule an appointment with your eye doctor.</span>
                  </div>
                </li>
                <li className="flex items-start group">
                  <span className="w-3 h-3 bg-red-500 rounded-full mr-3 mt-2 group-hover:scale-150 transition-transform"></span>
                  <div>
                    <span className="font-semibold text-gray-900">Severe:</span> 
                    <span className="ml-2">Advanced stage. Immediate medical attention recommended.</span>
                  </div>
                </li>
                <li className="flex items-start group">
                  <span className="w-3 h-3 bg-red-700 rounded-full mr-3 mt-2 group-hover:scale-150 transition-transform"></span>
                  <div>
                    <span className="font-semibold text-gray-900">Proliferate DR:</span> 
                    <span className="ml-2">Most severe stage. Urgent consultation required.</span>
                  </div>
                </li>
              </ul>
            </div>

            <div className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-2xl shadow-xl p-8 border border-amber-100 hover:shadow-2xl transition-all transform hover:-translate-y-1">
              <div className="flex items-center mb-6">
                <div className="w-16 h-16 bg-gradient-to-br from-amber-500 to-orange-600 rounded-2xl flex items-center justify-center mr-4 shadow-lg">
                  <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                </div>
                <h3 className="text-2xl font-bold text-gray-900">Important Information</h3>
              </div>
              <div className="space-y-4">
                <div className="bg-white/70 backdrop-blur-sm rounded-xl p-5 border border-amber-200">
                  <p className="text-gray-800 leading-relaxed mb-3">
                    <span className="font-bold text-amber-700">This AI tool is for screening purposes only</span> and should not replace professional medical diagnosis.
                  </p>
                  <p className="text-gray-700 leading-relaxed">
                    Always consult with a qualified <span className="font-semibold text-gray-900">ophthalmologist</span> for accurate diagnosis, treatment recommendations, and ongoing care.
                  </p>
                </div>
                <div className="bg-white/70 backdrop-blur-sm rounded-xl p-5 border border-blue-200">
                  <div className="flex items-start">
                    <svg className="w-6 h-6 text-blue-600 mr-3 mt-1 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <p className="text-gray-700 text-sm leading-relaxed">
                      Our AI model has been trained on thousands of retinal images and achieves high accuracy, but it's designed to assist, not replace, medical professionals.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* How It Works Section */}
        {!result && (
          <div className="mt-12 bg-gradient-to-r from-purple-50 via-pink-50 to-blue-50 rounded-3xl shadow-xl p-10 border border-purple-100">
            <h2 className="text-3xl font-bold text-center text-gray-900 mb-10">
              How It Works
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <div className="text-center group">
                <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg transform group-hover:scale-110 transition-transform">
                  <span className="text-3xl font-bold text-white">1</span>
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-2">Upload Image</h3>
                <p className="text-gray-600">Simply drag and drop or select your retinal scan image</p>
              </div>
              <div className="text-center group">
                <div className="w-20 h-20 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg transform group-hover:scale-110 transition-transform">
                  <span className="text-3xl font-bold text-white">2</span>
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-2">AI Analysis</h3>
                <p className="text-gray-600">Our advanced AI analyzes the image in seconds</p>
              </div>
              <div className="text-center group">
                <div className="w-20 h-20 bg-gradient-to-br from-purple-500 to-pink-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg transform group-hover:scale-110 transition-transform">
                  <span className="text-3xl font-bold text-white">3</span>
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-2">Get Results</h3>
                <p className="text-gray-600">Receive detailed analysis with confidence scores</p>
              </div>
            </div>
          </div>
        )}
      </main>
      <Footer />
    </div>
  );
}

export default App;
