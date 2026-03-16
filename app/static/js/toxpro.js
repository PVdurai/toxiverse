var http = new XMLHttpRequest();

function getResponseFromURL(queryUrl) {
  http.open('GET', queryUrl, false);
  http.send(null);
  if (http.status === 200) {
    return http.responseText;
  } else {
    return null;
  }
}

function updateOverview(data) {

    var totalChemicals = data.results.actives + data.results.inactives;
    var act_string = `${data.results.actives} (${Math.round(data.results.actives/totalChemicals*100, 2)}% of chemicals)`;
    var inact_string = `${data.results.inactives} (${Math.round(data.results.inactives/totalChemicals*100, 2)}% of chemicals)`;
    
    $("#num_act_text").text(act_string);
    $("#num_inact_text").text(inact_string);

}

function updateDataset() {
    var e = $('select[name="dataset-selection"]')[0];
    var currentDataset = e.options[e.selectedIndex].value;

    var hostname = location.protocol + '//' + location.host;
    var queryUrl =  hostname + "/api/datasets/" + currentDataset;
    var dataset_data = JSON.parse(getResponseFromURL(queryUrl));

    updateOverview(dataset_data);
    // plotBar(dataset_data.actives, dataset_data.inactives);
    // updateCompoundTable(dataset_data);

}

function getDatasetSelection() {
    var e = $('select[name="dataset-selection"]')[0];
    var currentDataset = e.options[e.selectedIndex].value;

    return currentDataset
}

function getProfileSelection() {
    var e = $('select[name="bioprofile-selection"]')[0];
    var currentProfile = e.options[e.selectedIndex].value;

    return currentProfile
}

function getPredictionSelection() {
    var e = $('select[name="prediction-selection"]')[0];
    var currentPrediction = e.options[e.selectedIndex].value;

    return currentPrediction
}

function getEndpointSelection() {
    var e = $('select[name="endpoint-selection"]')[0];
    var currentEndpoint = e.options[e.selectedIndex].value;

    return currentEndpoint
}

function updateEndpoint() {
    var e = $('select[name="endpoint-selection"]')[0];
    var currentEndpoint = e.options[e.selectedIndex].value;

    return currentEndpoint
}