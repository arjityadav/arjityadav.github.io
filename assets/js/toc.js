document.addEventListener("DOMContentLoaded", function () {
    var tocList = document.getElementById("toc-list");
    var headings = document.querySelectorAll("h2, h3"); // Select H2 & H3 headings
  
    headings.forEach(function (heading, index) {
      var headingText = heading.innerText.trim().toLowerCase(); // Normalize text for comparison
      // Exclude specific headings (adjust text as needed)
      if (headingText.includes("table of contents") || headingText.includes("nikhil bhavikatti")) {
        return; // Skip adding this heading
      }
  
      // Create a unique ID for each heading
      var anchorId = "section-" + index;
      heading.setAttribute("id", anchorId);
  
      // Create the TOC link
      var listItem = document.createElement("li");
      listItem.innerHTML = `<a href="#${anchorId}">${heading.innerText}</a>`;
      
      // Add class for styling
      if (heading.tagName === "H3") {
        listItem.classList.add("toc-subitem"); // Style subitems differently
      }
  
      tocList.appendChild(listItem);
    });
  });
