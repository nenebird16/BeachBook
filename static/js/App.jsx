
import React, { useState, useEffect } from 'react';
import ForceGraph from './components/ForceGraph';
import QueryPanel from './components/QueryPanel';
import DocumentUpload from './components/DocumentUpload';

export default function App() {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [queryResult, setQueryResult] = useState('');

  useEffect(() => {
    fetchGraphData();
  }, []);

  const fetchGraphData = async () => {
    const response = await fetch('/graph');
    const data = await response.json();
    setGraphData(data);
  };

  return (
    <div className="container-fluid">
      <div className="row">
        <div className="col-md-8">
          <div className="card">
            <div className="card-header">Knowledge Graph</div>
            <div className="card-body">
              <ForceGraph data={graphData} />
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <DocumentUpload onUpload={fetchGraphData} />
          <QueryPanel onQueryResult={setQueryResult} />
          <div className="card mt-3">
            <div className="card-header">Query Result</div>
            <div className="card-body">
              <div id="queryResult">{queryResult}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
