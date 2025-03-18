
import React, { useState } from 'react';

export default function QueryPanel({ onQueryResult }) {
  const [query, setQuery] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    const response = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });
    const data = await response.json();
    onQueryResult(data.response);
  };

  return (
    <div className="card mt-3">
      <div className="card-header">Query Knowledge Base</div>
      <div className="card-body">
        <form onSubmit={handleSubmit}>
          <div className="mb-3">
            <textarea
              className="form-control"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter your query..."
              rows="3"
            />
          </div>
          <button type="submit" className="btn btn-primary">Submit Query</button>
        </form>
      </div>
    </div>
  );
}
