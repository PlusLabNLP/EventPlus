
// two select box
var news_array = [
    "The other customers fled, and the police said it did not appear that anyone else was injured",
    "A powerful ice storm continues to maintain its grip. Yesterday New York governor George Pataki toured five counties that have been declared under a state of emergency",
    "Despite the recent possibility of military conflict with Iraq, oil prices have been falling, that's because of a worldwide glut of oil"
];

var bio_array = [
    "We have found that the HTLV-1 transactivator protein, tax, acts as a costimulatory signal for GM-CSF and IL-2 gene transcription, in that it can cooperate with TCR signals to mediate high level gene expression.",
    "We show that ligand-induced homodimerization of chimeric surface receptors consisting of the extracellular and transmembrane domains of the erythropoietin receptor and of the intracellular domain of IL-4Ralpha induces Janus kinase 1 (Jak1) activation, STAT6 activation, and Cepsilon germline transcripts in human B cell line BJAB."
];

String.prototype.template = function() {
var args = arguments;
return this.replace(/\{(\d+)\}/g,function(m,i){return args[i];});
};

function htmlEntities(str) {
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39');
}

function select_topic() {
   var val = document.getElementById("topic").value;
   var btnelem = document.getElementById("analyze-text-btn");

   if (val == "news") {
       btnelem.disabled = false;
       console.log(btnelem.disabled);
        var news = "";
        news += "<option value=''>---</option>";
        for(let i = 0; i < news_array.length; i++) {
            news += "<option value='{0}'>".template(htmlEntities(news_array[i])) + htmlEntities(news_array[i]).substring(0, 30) + "..." + "</option>"
        }
        // console.log("news", news)
        $("#examples").html(news);
    }
    else if (val == "bio") {
       btnelem.disabled = false;
       var bio = "";
       bio += "<option value=''>---</option>";
        for(let i = 0; i < bio_array.length; i++) {
            bio += "<option value='{0}'>".template(htmlEntities(bio_array[i])) + htmlEntities(bio_array[i]).substring(0, 30) + "..." + "</option>"
        }
        $("#examples").html(bio);
    }
}

function give_examples() {
    var e2 = document.getElementById("examples");
    document.getElementById("analyze-text-input").value = e2.value
}
