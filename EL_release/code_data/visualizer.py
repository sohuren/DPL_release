import random

def make_html_file(examples, filename, class_entity):
    """
    Visualizes attention of a model in a HTML file.
    :param examples:
    :param filename:
    :return:
    """

    def attention_to_rgb(attention):
        
        # red = int(attention * 255)
        # green = int(255 - red)
        red = 255
        green = int(255 * (1-attention))
        blue = int(255 * (1-attention))

        return 'rgb(%s,%s,%s)' % (str(red), str(green), blue)
    
    def idx_to_interval(idx, match):    
        for item in match:
            if idx >= item[0] and idx <= item[1]:
               return True
        return False
    
    def idx_to_key(idx, match):    
        for item in match:
            if idx >= item[0] and idx <= item[1]:
               return item
    
    def write_context(out_file, context, matches, class_entity, example_index):

    # for each word in the context 
        for idx, word in enumerate(context):    
        # check if it's classified as current class
            current_class = idx_to_interval(idx, matches[class_entity])
            other_class = False; category=""
            for key in matches.keys():
                if key != class_entity:
                   other_class = idx_to_interval(idx, matches[key])
                   if other_class:
                        category = key                            
                        break
            
            # match[2][1] is the predicted label
            if current_class:
                match = idx_to_key(idx, matches[class_entity])
                pred = match[2]

            # checking whether they overlapping or not, only displace the class_entity
            if current_class and not other_class:
                out_file.write('<u>')
                out_file.write('<mark class="g%(class)s" cursor="pointer" title="%(pointer_text)s" style="background-color:%(rgb)s;">%(word)s </mark>' %
                           {'pointer_text' :class_entity+":"+ str(pred), 'rgb' : attention_to_rgb(0.5), 'word' : word, 'class' : str(example_index) + "-" + word.replace("@", "")})
                out_file.write('</u>')
            elif current_class and other_class:
                out_file.write('<u>')
                out_file.write('<mark class="g%(class)s" cursor="pointer" title="%(pointer_text)s" style="background-color:%(rgb)s;">%(word)s </mark>' %
                           {'pointer_text' :class_entity+category+str(pred), 'rgb' : attention_to_rgb(1), 'word' : word, 'class' : str(example_index) + "-" + word.replace("@", "")})
                out_file.write('</u>')
            else:
                out_file.write('<mark class="g%(class)s" cursor="pointer" title="%(pointer_text)s" style="background-color:%(rgb)s;">%(word)s </mark>' %
                           {'pointer_text' : "no matching", 'rgb' : attention_to_rgb(0), 'word' : word, 'class' : str(example_index) + "-" + word.replace("@", "")})


    out_file = open(filename, 'w')
    out_file.write('''<!DOCTYPE html>
                    <head>
                    <link rel="stylesheet" href="//code.jquery.com/ui/1.10.4/themes/smoothness/jquery-ui.css" />
                    <script src="http://ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min.js"></script>
                    <script src="http://code.jquery.com/ui/1.10.4/jquery-ui.js"></script>
                    </head><html>
                    <body><h1>Awesome Network Output</h1>
                    <script>
                    function highlight(x)
                    {
                        //alert(x)
                        $("." + x).addClass('foo')
                        //$(".foo").css({'font-weight': 600})
                        $(".foo").animate({
                            color: "blue"
                            }, {duration: 200} );
                    }
                    function unhighlight()
                    {
                        $(".foo").css({'font-weight': "normal"})
                        $(".foo").animate({
                            color: "black"
                            }, {duration: 200} );
                        $(".foo").removeClass('foo')
                    }
                    </script>
    
                    ''')
                                
    example_idx = 0

    for trial_key in examples.keys():
                
        # this is for single instance
        out_file.write("<p>")
        out_file.write("<p><b> %s </b></p>"%(trial_key))
        out_file.write("<p><b> inclusion </b></p>")
        for instance in examples[trial_key]['inc']:
            context_inc, matches_inc = instance
            write_context(out_file, context_inc, matches_inc, class_entity, example_idx)
            out_file.write("<p></p>")

        out_file.write("<p><b> exclusion </b></p>")
        for instance in examples[trial_key]['exc']:
            context_exc, matches_exc = instance
            write_context(out_file, context_exc, matches_exc, class_entity, example_idx) 
            out_file.write("<p></p>")
            
        out_file.write("</p>")
        example_idx += 1

    out_file.write('</body></html>')
    out_file.close()



def make_html_file_confidence(examples, filename, gene_key):
    """
    Visualizes attention of a model in a HTML file.
    :param examples:
    :param filename:
    :return:
    """

    def attention_to_rgb(attention):
        
        # red = int(attention * 255)
        # green = int(255 - red)
        red = 255
        green = int(255 * (1-attention))
        blue = int(255 * (1-attention))

        return 'rgb(%s,%s,%s)' % (str(red), str(green), blue)
    
    def idx_to_interval(idx, match):    
        for item in match:
            if idx >= item[0] and idx <= item[1]:
                return True
        return False
    
    def idx_to_key(idx, match):    
        for item in match:
            if idx >= item[0] and idx <= item[1]:
                return item
    
    def write_context(out_file, context, matches, example_index):

        # for each word in the context 
        for idx, word in enumerate(context):
            # check if it's classified as current class
            current_class = idx_to_interval(idx, [matches])
            if current_class:
                pred = matches[2] # this is the current predicted label

            # checking whether they overlapping or not, only displace the class_entity
            if " ".join(context[matches[0]:matches[1]+1]) in gene_key.keys():
                id = gene_key[" ".join(context[matches[0]:matches[1]+1])]
            else:
                id = ''        
            if current_class:
                out_file.write('<u>')
                out_file.write('<mark class="g%(class)s" cursor="pointer" title="%(pointer_text)s" style="background-color:%(rgb)s;">%(word)s </mark>' %
                           {'pointer_text' : id, 'rgb' : attention_to_rgb(1), 'word' : word, 'class' : str(example_index) + "-" + word.replace("@", "")})
                out_file.write('</u>')
            else:
                out_file.write('<mark class="g%(class)s" cursor="pointer" title="%(pointer_text)s" style="background-color:%(rgb)s;">%(word)s </mark>' %
                           {'pointer_text' : "no matching", 'rgb' : attention_to_rgb(0), 'word' : word, 'class' : str(example_index) + "-" + word.replace("@", "")})


    out_file = open(filename, 'w')
    out_file.write('''<!DOCTYPE html>
                    <head>
                    <link rel="stylesheet" href="//code.jquery.com/ui/1.10.4/themes/smoothness/jquery-ui.css" />
                    <script src="http://ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min.js"></script>
                    <script src="http://code.jquery.com/ui/1.10.4/jquery-ui.js"></script>
                    </head><html>
                    <body><h1>Awesome Network Output</h1>
                    <script>
                    function highlight(x)
                    {
                        //alert(x)
                        $("." + x).addClass('foo')
                        //$(".foo").css({'font-weight': 600})
                        $(".foo").animate({
                            color: "blue"
                            }, {duration: 200} );
                    }
                    function unhighlight()
                    {
                        $(".foo").css({'font-weight': "normal"})
                        $(".foo").animate({
                            color: "black"
                            }, {duration: 200} );
                        $(".foo").removeClass('foo')
                    }
                    </script>
    
                    ''')
                                
    example_idx = 0
    out_file.write("<p><H2> %s </H2></p>"%("Positive Example"))    
    for instance in examples['positive']:
                
        # this is for single instance
        trial = instance[0]
        context = instance[1]
        item = instance[2]
        out_file.write("<p>")
        out_file.write("<p><b> %s </b></p>"%(str(example_idx)))
        out_file.write("<p><b> %s </b></p>"%(trial))
        out_file.write("<p><b> 0: %f 1: %f </b></p>"%(item[3][0][0], item[3][0][1]))
        write_context(out_file, context, item, example_idx)
        out_file.write("</p>")
        example_idx += 1


    out_file.write("<p><H2> %s </H2></p>"%("Negative Example"))    
    example_idx = 0

    for instance in examples['negative']:
                
        # this is for single instance
        trial = instance[0]
        context = instance[1]
        item = instance[2]
        out_file.write("<p>")
        out_file.write("<p><b> %s </b></p>"%(str(example_idx)))
        out_file.write("<p><b> %s </b></p>"%(trial))
        out_file.write("<p><b> 0: %s 1: %s </b></p>"%(item[3][0][0], item[3][0][1]))
        write_context(out_file, context, item, example_idx)
        out_file.write("</p>")
        example_idx += 1

    out_file.write('</body></html>')
    out_file.close()
    

'''
if __name__ == "__main__":

    examples = []

    for i in range(4):
        context = 'Ruda met Alice . Bob met Ruda . Alice met Ruda .'.split(' ')
        matches = {}
        matches['drug'] = [(1,2), (3,5)]
        matches['gene'] = [(6,8)]
        matches['disease'] = []
        examples.append((context, matches))

    make_html_file(examples, 'output.html')
'''
