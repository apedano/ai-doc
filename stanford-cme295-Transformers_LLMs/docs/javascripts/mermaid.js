document$.subscribe(function () {
    mermaid.initialize({
        startOnLoad: true
    });
    console.log("Mermaid loaded")
    mermaid.run();
});