
class JsonBuilder:
    def __init__(self, B2I_trigger, B2I_argument, B2I_ner):
        self.B2I_trigger = B2I_trigger
        self.B2I_argument = B2I_argument
        self.B2I_ner = B2I_ner

    def iob_to_obj(self, y, B2I):
        '''
        B2I : {'B-AGENT': 'I-AGENT', 'B-PATIENT': 'I-PATIENT'}
        '''
        obj = []
        in_obj = False
        curr_obj = []
        curr_I = None
        for i in range(len(y)):
            # end of obj
            if in_obj:
                if y[i] != curr_I:
                    obj.append(curr_obj + [i-1])
                    curr_obj = []
                    curr_I = None
                    in_obj = False
                else:
                    if i == len(y) - 1:
                        obj.append(curr_obj + [i])
            # beginning of obj
            if y[i] in B2I:
                curr_obj = [y[i][2:], i]
                curr_I = B2I[y[i]]
                in_obj = True
                if i == len(y) - 1:
                    obj.append(curr_obj + [i])
        return obj
    def from_preds(self, input_sent, y_preds_t, y_preds_e, y_preds_ner):
        assert len(y_preds_t) == len(y_preds_e)
        preds = []
        for y_pred_t, y_pred_e in zip(y_preds_t, y_preds_e):
            preds.append({
                'trigger': y_pred_t,
                'argument': y_pred_e
                })
        ner = self.iob_to_obj(y_preds_ner[0], self.B2I_ner)
        ner = [[x[1], x[2], x[0]] for x in ner]  # convert the order for each ner obj
        out = []
        events_pred = self.convert_out_dicts_to_event_dicts(preds, input_sent)
        out.append({'tokens': input_sent,
                    'events': events_pred,
                    'ner': ner
            })
        return out


    def convert_out_dicts_to_event_dicts(self, sel_preds, input_sent):
        '''
        `sel_preds` contain sent-level prediction
        return a list of dicts, which will be used to create the BetterEvent objs
        `data_type`, currently support choose from ['local', 'ssvm']
        '''

        out_dicts = []
        cnt = 1
        for event in sel_preds:
            out_dict = {}
            # sent_id = event['sent_id']

            tri_seq = event['trigger']
            trigger_objs = self.iob_to_obj(tri_seq, self.B2I_trigger)
            if len(trigger_objs) == 0:
                continue
            else:

                event_type = trigger_objs[0][0]
                out_dict['event_type'] = event_type
                trigger_span_dicts = self.get_span_dicts_from_objs(trigger_objs, input_sent, task='trigger')
                arg_objs = self.iob_to_obj(event['argument'], self.B2I_argument)
                argu_span_dicts = self.get_span_dicts_from_objs(arg_objs, input_sent, task='argument')
                out_dict['triggers'] = trigger_span_dicts

                out_dict['arguments'] = argu_span_dicts

            cnt += 1
            out_dicts.append(out_dict)
        return out_dicts

    def get_span_dicts_from_objs(self, objs, input_sent, task='trigger'):
        span_dicts = []
        for obj in objs:
            role = obj[0]
            l_idx = obj[1]
            r_idx = obj[2]
            text = input_sent[l_idx] if r_idx == l_idx \
                    else ' '.join(input_sent[l_idx:r_idx+1])
            if task == 'trigger':
                span_dict = {'event_type': role,
                             'text': text,
                             'start_token': l_idx,
                             'end_token': r_idx
                        }
            elif task == 'argument':
                span_dict = {'role': role,
                             'text': text,
                             'start_token': l_idx,
                             'end_token': r_idx
                        }
            span_dicts.append(span_dict)
        return span_dicts
