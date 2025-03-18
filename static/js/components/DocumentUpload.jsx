
import React from 'react';

export default function DocumentUpload({ onUpload }) {
  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('file', e.target.documentFile.files[0]);

    const response = await fetch('/upload', {
      method: 'POST',
      body: formData
    });
    
    if (response.ok) {
      alert('Document uploaded successfully!');
      onUpload();
    } else {
      const data = await response.json();
      alert(`Error: ${data.error}`);
    }
  };

  return (
    <div className="card">
      <div className="card-header">Upload Document</div>
      <div className="card-body">
        <form onSubmit={handleSubmit}>
          <div className="mb-3">
            <input
              type="file"
              className="form-control"
              id="documentFile"
              required
            />
          </div>
          <button type="submit" className="btn btn-primary">Upload</button>
        </form>
      </div>
    </div>
  );
}
