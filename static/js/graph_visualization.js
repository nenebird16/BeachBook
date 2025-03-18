function updateGraph() {
    const width = document.getElementById('graphVisualization').clientWidth;
    const height = 500;

    // Clear existing visualization
    d3.select("#graphVisualization").html("");

    // Create SVG container
    const svg = d3.select("#graphVisualization")
        .append("svg")
        .attr("width", width)
        .attr("height", height);

    // Create force simulation
    const simulation = d3.forceSimulation()
        .force("link", d3.forceLink().id(d => d.id))
        .force("charge", d3.forceManyBody().strength(-100))
        .force("center", d3.forceCenter(width / 2, height / 2));

    // Fetch and render graph data
    fetch('/graph')
        .then(response => response.json())
        .then(graph => {
            // Create links
            const link = svg.append("g")
                .selectAll("line")
                .data(graph.links)
                .enter().append("line")
                .attr("stroke", "#999")
                .attr("stroke-opacity", 0.6);

            // Create nodes
            const node = svg.append("g")
                .selectAll("circle")
                .data(graph.nodes)
                .enter().append("circle")
                .attr("r", 5)
                .attr("fill", d => d.label === "Document" ? "#4CAF50" : "#2196F3")
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));

            // Add tooltips
            node.append("title")
                .text(d => d.properties.title || d.properties.name);

            // Update simulation
            simulation
                .nodes(graph.nodes)
                .on("tick", ticked);

            simulation.force("link")
                .links(graph.links);

            function ticked() {
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);

                node
                    .attr("cx", d => d.x)
                    .attr("cy", d => d.y);
            }
        });

    function dragstarted(event) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
    }

    function dragged(event) {
        event.subject.fx = event.x;
        event.subject.fy = event.y;
    }

    function dragended(event) {
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
    }
}
