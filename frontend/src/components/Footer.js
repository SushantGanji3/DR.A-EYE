import React from 'react';

function Footer() {
  return (
    <footer className="bg-gray-900 text-gray-300 mt-auto">
      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <h3 className="text-white font-bold text-lg mb-4">DR.A-EYE</h3>
            <p className="text-gray-400 text-sm leading-relaxed">
              Advanced AI-powered diabetic retinopathy screening tool using deep learning technology 
              for early detection and prevention.
            </p>
          </div>
          <div>
            <h4 className="text-white font-semibold mb-4">Technology</h4>
            <ul className="space-y-2 text-sm text-gray-400">
              <li>• ResNet-18 Deep Learning</li>
              <li>• Computer Vision</li>
              <li>• Medical Image Analysis</li>
              <li>• Real-time Processing</li>
            </ul>
          </div>
          <div>
            <h4 className="text-white font-semibold mb-4">Disclaimer</h4>
            <p className="text-gray-400 text-sm leading-relaxed">
              This tool is for screening purposes only. Always consult with a qualified 
              ophthalmologist for professional medical diagnosis and treatment.
            </p>
          </div>
        </div>
        <div className="border-t border-gray-800 mt-8 pt-6 text-center text-sm text-gray-500">
          <p>© 2025 DR.A-EYE. Powered by Deep Learning & Computer Vision.</p>
        </div>
      </div>
    </footer>
  );
}

export default Footer;

