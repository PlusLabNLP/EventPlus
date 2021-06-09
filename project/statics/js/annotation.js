const intro = introJs();
intro.setOptions({
    exitOnOverlayClick:false,
    steps: [
        {intro: "Welcome to EventPlus! Let\'s take a tour!"},
        {
            element: "#btn-feature",
            intro: "Please take a look at features in EventPlus"
        },
        {
            element: "#topic",
            intro: "Select the domain of your input, EventPlus support news domain and biomedical domain!"
        },
        {
            element: "#examples",
            intro: "Select a sentence or input your sentences below."
        },
        {
            element: "#analyze-text-btn",
            intro: "Click on analyze text button!"
        },
        {
            element: "#show_annotation",
            intro: "We visualize entities in your input and their NER labels, they will be candidate arguments for your event!"
        },
        {
            element: "#displayEvents",
            intro: "Here are all events that we extracted! Please click on any of them to see their corresponding arguments!"
        },
        {
            element: "#show_annotation",
            intro: "We visualize your event and its corresponding arguments here!"
        },
        {
            element: "#graph",
            intro: "Temporal relations between events if there are any and duration of events as node labels"
        }
    ]
});
intro.start();

var return_value;
var labels;
var tokens;
var mark_default = "padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone";
var span_default = "font-style: italic; background: #f4db4b; font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem";
var ner_default = "font-style: italic; background: #f4c2c2; font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem";
var default_annotation;
String.prototype.template = function() {
    var args = arguments;
    return this.replace(/\{(\d+)\}/g,function(m,i){return args[i];});
};


$( function(){
    $( document ).on('click', '#analyze-text-btn', function(e){
        var input_text = $('#analyze-text-input').val().toString();
        console.log('input_text:', input_text);
        var topic = $('#topic').val().toString();
        ajaxAnalyzeText(input_text, topic);
    });

});

function ajaxAnalyzeText (input_text, topic){
    console.log("ajax start");
    $body = $("body");
    $body.addClass("loading");
    $.ajax({
        url: '/analyze_text/',
        type: 'post',
        async: true,
        dataType: 'json',
        data: {
            text: input_text,
            domain: topic,
        },
        beforeSend: function (xhr, settings) {
            if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
                xhr.setRequestHeader("X-CSRFToken", csrftoken);
            }
        },
        success: function (data) {
            console.log("ajax end");
            $body.removeClass("loading");
            return_value = data;
            $("#graph").empty();
            draw_graph(return_value);
            tokens = return_value.tokens;
            labels = return_value.labels;
            var annotation = "";
            var event_display = '<label>events:</label>';
            // what if we delete the maintained visitedList to get rid of the nested problem.
            // var visitedList = [];
            // find all start indexes for ners and all start indexes for triggers
            var ner = [];
            var triggers = new Set();
            for (i = 0; i < labels.length; i++) {
                if (labels[i].role === "trigger") {
                    triggers.add(labels[i].start)
                }
                if ("ner" in labels[i]) {
                    ner[labels[i].start] = labels[i]
                }
            }
            var visitListNer = [];
            for (let i = 0; i < tokens.length; i++) {
                // console.log(i.toString() + ": " + visitedList);
                // if (visitedList.includes(i)) {
                //     continue;
                // }
                if (triggers.has(i)) {
                    labels.forEach(function (item, index) {
                        if ((item.start === i) && (item.role === "trigger")) {
                            var text = " ";
                            for (index = i; index < item.end; index++) {
                                // this is correct
                                var visitedList = [];
                                if (visitedList.includes(index)) {
                                    continue;
                                }
                                text += tokens.slice(index, index + 1);
                                text += " ";
                                visitedList.push(index);
                            }
                            var mark_style = "background:" + item.color + ";" + mark_default;
                            event_display += "<span style='{0}; cursor: pointer' eventId='{1}' onclick='event_click(this)'>{2}</span>".template(mark_style, item.event, text);
                        }
                    });
                }

                if (i in ner) {
                    var label = ner[i]
                    var text = " ";
                    for (index = i; index < label.end; index++) {
                        if(visitListNer.includes(index)) {
                            continue;
                        }
                        text += tokens.slice(index, index + 1);
                        text += " ";
                        visitListNer.push(index)
                    }
                    ner_label = label.ner;
                    annotation += "<mark style='text-decoration-line: underline; text-decoration-style: wavy;'>{0}</mark><span style='{1}' position>{2}</span></mark>&nbsp;".template(text, ner_default, ner_label)
                }
                if (!(i in ner) && !(i in triggers) && !(visitListNer.includes(i))) {
                    annotation += tokens[i];
                    annotation += " ";
                }
            }
           $("#show_annotation").html(annotation)
            // give a default annotation as the annotation when it first loads
            default_annotation = annotation;
            $("#displayEvents").html(event_display)
        }
    });
}

function event_click(obj) {
    var clicks = $(this).data('clicks');
    if (clicks) {
         // odd clicks
        // console.log("odd clicks")
        onlyPlotThis(obj.getAttribute("eventId"));
    } else {
         // even clicks
        // console.log("even clicks")
         $("#show_annotation").html(default_annotation)

    }
      $(this).data("clicks", !clicks);
}

function onlyPlotThis(eventId) {

    var annotation = "";
    var visitedList = [];
    var event_display = "";
    var this_event = {};
    // find everything corresponding to this event and mark it as a dictionary
    labels.forEach(function (item, index) {
        if (item.event == eventId) {
                this_event[item.start] = item
            }
        });

    for (let i=0; i < tokens.length; i++) {
        // console.log(i.toString() + ": " + visitedList);
         if (visitedList.includes(i)) {
            continue;
        }
        if (i in this_event && !(visitedList.includes(i))) {
            var label = this_event[i];
            var text = " ";
            for(index = i; index < label.end; index++) {
                text += tokens.slice(index, index + 1);
                text += " ";
                visitedList.push(index);
            }
            if ("event" in label && label.event == eventId) {
                  if (label.role === "trigger") {
                    // this labeled item is a trigger
                    var mark_style = "background:" + label.color + ";" + mark_default;
                    annotation += "<mark style='{0}'>{1}<span style='{2}'>{3}</span></mark>".template(mark_style, text, span_default, label.label);
                    event_display += "<span style='{0}; cursor: pointer' eventId='{1}' onclick='event_click(this)'>{2}</span>".template(mark_style, label.event, text);
                } else if (label.role === "argument") {
                      // this label item is an argument
                      var mark_style = "background:" + label.color + ";" + mark_default;
                      if("ner" in label) {
                        mark_style += "text-decoration-line: underline;";
                        ner_label = label.ner;
                        annotation += "<mark style='{0}'>{1}<span style='{2}'>{3}</span><span style='{4}'>{5}</span></mark>".template(mark_style, text, span_default, label.label, ner_default, ner_label);
                      } else {
                          annotation += "<mark style='{0}'>{1}<span style='{2}'>{3}</span></mark>".template(mark_style, text, span_default, label.label);
                      }
                }
            } else {
                annotation += text;
                annotation += " ";
            }
        } else {
            annotation += tokens[i];
            annotation += " ";
        }
    }
    $("#show_annotation").html(annotation)

}

// var labels = return_value.labels

// for(let i = 0; i < cherry_obj.tokens.length; i++){
//
// }
//
// for(let i = 0; i < cherry_obj.events.length; i++) {
//     var mark_style = "background:" + "#42a4f0;" + mark_default
//     $("#displayEvents").html("<mark style='{0}'>{1}</mark>".template(mark_style, cherry_obj.events[i].triggers.text))
// }


