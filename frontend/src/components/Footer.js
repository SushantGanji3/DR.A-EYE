import React from 'react';

function Footer() {
  return (
    <footer className="bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-gray-300 mt-auto">
      <div className="container mx-auto px-4 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-10">
          <div>
            <div className="flex items-center mb-5">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center mr-3">
                <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
              </div>
              <h3 className="text-2xl font-bold text-white">DR.A-EYE</h3>
            </div>
            <p className="text-gray-400 leading-relaxed">
              Advanced AI-powered diabetic retinopathy screening tool using cutting-edge deep learning technology 
              for early detection and prevention. Your health is our priority.
            </p>
          </div>
          <div>
            <h4 className="text-white font-bold text-lg mb-5 flex items-center">
              <span className="mr-2">🔬</span>
              Technology
            </h4>
            <ul className="space-y-3 text-gray-400">
              <li className="flex items-center hover:text-white transition-colors">
                <span className="w-2 h-2 bg-blue-500 rounded-full mr-3"></span>
                ResNet-18 Deep Learning
              </li>
              <li className="flex items-center hover:text-white transition-colors">
                <span className="w-2 h-2 bg-indigo-500 rounded-full mr-3"></span>
                Computer Vision AI
              </li>
              <li className="flex items-center hover:text-white transition-colors">
                <span className="w-2 h-2 bg-purple-500 rounded-full mr-3"></span>
                Medical Image Analysis
              </li>
              <li className="flex items-center hover:text-white transition-colors">
                <span className="w-2 h-2 bg-pink-500 rounded-full mr-3"></span>
                Real-time Processing
              </li>
            </ul>
          </div>
          <div>
            <h4 className="text-white font-bold text-lg mb-5 flex items-center">
              <span className="mr-2">⚠️</span>
              Important Notice
            </h4>
            <div className="bg-gray-800/50 rounded-xl p-5 border border-gray-700">
              <p className="text-gray-300 leading-relaxed text-sm">
                This AI-powered tool is designed for <strong className="text-white">screening purposes only</strong> and should not replace professional medical diagnosis.
              </p>
              <p className="text-gray-300 leading-relaxed text-sm mt-3">
                Always consult with a qualified <strong className="text-white">ophthalmologist</strong> for professional medical diagnosis, treatment recommendations, and ongoing care.
              </p>
            </div>
          </div>
        </div>
        <div className="border-t border-gray-700 mt-10 pt-8 text-center">
          <p className="text-gray-400 text-sm">
            © 2025 <span className="text-white font-semibold">DR.A-EYE</span>. Powered by Deep Learning & Computer Vision. 
            <span className="block mt-2">Made with ❤️ for better eye health</span>
          </p>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
