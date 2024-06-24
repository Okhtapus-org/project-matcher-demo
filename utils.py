def create_accordion_html(relevant_entries):
    accordion_html = """
    <style>
    .accordion {
      background-color: #f1f1f1;
      color: #444;
      cursor: pointer;
      padding: 18px;
      width: 100%;
      text-align: left;
      border: none;
      outline: none;
      transition: 0.4s;
      margin-bottom: 5px;
    }
    .active, .accordion:hover {
      background-color: #ccc;
    }
    .panel {
      padding: 0 18px;
      background-color: white;
      display: none;
      overflow: hidden;
    }
    </style>

    <script>
    function toggleAccordion(accordionId) {
      var panel = document.getElementById(accordionId);
      if (panel.style.display === "block") {
        panel.style.display = "none";
      } else {
        panel.style.display = "block";
      }
    }
    </script>
    """

    for i, (_, fellow) in enumerate(relevant_entries.iterrows()):
        accordion_html += f"""
        <button class="accordion" onclick="toggleAccordion('panel{i}')">{fellow['Name']} - {fellow['Role Title']}</button>
        <div id="panel{i}" class="panel">
          <p><strong>Bio:</strong> {fellow['Bio']}</p>
          <p><strong>Wants to engage by:</strong> {fellow['Wants to engage by']}</p>
          <p><strong>VB Priority area(s):</strong> {fellow['VB Priority area(s)']}</p>
          <p><strong>Sector/Type:</strong> {fellow['Sector/ Type']}</p>
          <p><strong>Spike:</strong> {fellow['Spike']}</p>
        </div>
        """

    return accordion_html