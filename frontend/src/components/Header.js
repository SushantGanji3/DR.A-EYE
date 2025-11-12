import React from 'react';

function Header() {
  return (
    <header className="bg-white shadow-md">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center">
              <span className="text-white font-bold text-xl">DR</span>
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-800">DR.A-EYE</h1>
              <p className="text-sm text-gray-600">Diabetic Retinopathy Screening Tool</p>
            </div>
          </div>
          <div className="text-sm text-gray-600">
            Powered by Deep Learning
          </div>
        </div>
      </div>
    </header>
  );
}

export default Header;

