import React, { useState, useRef } from 'react';

function ImageUpload({ onPredict, loading, error }) {
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileSelect = (file) => {
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file (PNG, JPG, or JPEG)');
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      alert('Image size should be less than 10MB. Please compress or select a smaller image.');
      return;
    }

    setSelectedImage(file);
    
    // Create preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const handleFileInput = (e) => {
    const file = e.target.files[0];
    handleFileSelect(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    handleFileSelect(file);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!selectedImage) {
      alert('Please select an image first');
      return;
    }

    const formData = new FormData();
    formData.append('image', selectedImage);
    onPredict(formData);
  };

  const handleReset = () => {
    setSelectedImage(null);
    setPreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="w-full">
      <form onSubmit={handleSubmit} className="space-y-8">
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`relative border-3 border-dashed rounded-3xl p-16 text-center transition-all duration-300 ${
            isDragging
              ? 'border-blue-500 bg-gradient-to-br from-blue-100 to-indigo-100 scale-105 shadow-2xl'
              : preview
              ? 'border-gray-200 bg-gradient-to-br from-gray-50 to-blue-50/30'
              : 'border-blue-300 bg-gradient-to-br from-blue-50/50 via-indigo-50/30 to-purple-50/50 hover:border-blue-400 hover:bg-gradient-to-br hover:from-blue-100 hover:via-indigo-100 hover:to-purple-100 hover:shadow-xl'
          }`}
        >
          {preview ? (
            <div className="space-y-6 animate-fade-in">
              <div className="relative inline-block">
                <div className="absolute -inset-4 bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-400 rounded-3xl blur-2xl opacity-30"></div>
                <img
                  src={preview}
                  alt="Preview"
                  className="relative max-h-96 mx-auto rounded-2xl shadow-2xl border-4 border-white transform hover:scale-105 transition-transform"
                />
                <div className="absolute -top-3 -right-3 w-12 h-12 bg-gradient-to-br from-green-400 to-emerald-500 rounded-full flex items-center justify-center shadow-xl animate-bounce">
                  <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
              </div>
              <button
                type="button"
                onClick={handleReset}
                className="inline-flex items-center px-6 py-3 text-base font-semibold text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded-xl transition-all transform hover:scale-105"
              >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Choose Different Image
              </button>
            </div>
          ) : (
            <div className="space-y-8">
              <div className="mx-auto w-32 h-32 bg-gradient-to-br from-blue-200 via-indigo-200 to-purple-200 rounded-3xl flex items-center justify-center shadow-inner transform hover:scale-110 transition-transform">
                <svg
                  className="w-16 h-16 text-blue-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                  />
                </svg>
              </div>
              <div>
                <h3 className="text-3xl font-bold text-gray-900 mb-3">
                  {isDragging ? '🎉 Drop it here!' : '📸 Upload Your Retinal Scan'}
                </h3>
                <p className="text-xl text-gray-600 mb-8 font-medium">
                  {isDragging ? 'Release to upload' : 'Drag and drop your image, or click the button below'}
                </p>
                <label
                  htmlFor="image-upload"
                  className="cursor-pointer inline-flex items-center px-10 py-5 bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white rounded-2xl font-bold text-lg shadow-2xl hover:shadow-3xl hover:from-blue-700 hover:via-indigo-700 hover:to-purple-700 transform hover:scale-105 transition-all duration-200"
                >
                  <svg className="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  Choose Image File
                  <input
                    id="image-upload"
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileInput}
                    className="hidden"
                  />
                </label>
              </div>
              <div className="flex items-center justify-center space-x-8 text-base text-gray-600">
                <div className="flex items-center bg-white/80 backdrop-blur-sm px-4 py-2 rounded-full shadow-md">
                  <svg className="w-5 h-5 mr-2 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  PNG, JPG, JPEG
                </div>
                <div className="flex items-center bg-white/80 backdrop-blur-sm px-4 py-2 rounded-full shadow-md">
                  <svg className="w-5 h-5 mr-2 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
                  </svg>
                  Max 10MB
                </div>
                <div className="flex items-center bg-white/80 backdrop-blur-sm px-4 py-2 rounded-full shadow-md">
                  <svg className="w-5 h-5 mr-2 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Quick Analysis
                </div>
              </div>
            </div>
          )}
        </div>

        {selectedImage && (
          <div className="flex justify-center">
            <button
              type="submit"
              disabled={loading}
              className={`group relative inline-flex items-center px-12 py-5 bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white rounded-2xl font-bold text-xl shadow-2xl hover:shadow-3xl hover:from-blue-700 hover:via-indigo-700 hover:to-purple-700 transform hover:scale-105 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none ${
                loading ? 'opacity-50' : ''
              }`}
            >
              {loading ? (
                <>
                  <svg
                    className="animate-spin -ml-1 mr-4 h-7 w-7 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                  <span className="text-lg">Analyzing Your Image...</span>
                </>
              ) : (
                <>
                  <svg className="w-7 h-7 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>Analyze Image Now</span>
                  <svg className="w-5 h-5 ml-3 transform group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </>
              )}
            </button>
          </div>
        )}
      </form>
    </div>
  );
}

export default ImageUpload;
