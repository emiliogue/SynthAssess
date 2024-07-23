import plotly.io as pio
from io import BytesIO
import base64
import pandas as pd

class ReportGenerator:
    def __init__(self, report_title="SynthAssess Report"):
        self.report_title = report_title
        self.sections = {
            'Data Overview': {},
            'Statistical Similarity': {},
            'Privacy': {},
            'ML Efficacy': {}
        }
        self.icons = {
            'Data Overview': '📊',
            'Statistical Similarity': '📈',
            'Privacy': '🔒',
            'ML Efficacy': '🎯'
        }

    def add_subsection(self, section, subsection):
        '''
        Add a subsection to a section in the report.
        
        param: section: str: name of the section
        param: subsection: str: name of the subsection
        '''
        if section in self.sections:
            self.sections[section][subsection] = []
        else:
            print(f"Section '{section}' does not exist.")

    def add_text_to_section(self, section, text, subsection=None):
        '''
        Add text content to a section in the report.

        param: section: str: name of the section
        param: text: str: text content to add
        param: subsection: str: name of the subsection(optional)
        '''
        if section in self.sections:
            if subsection:
                if subsection not in self.sections[section]:
                    self.add_subsection(section, subsection)
                self.sections[section][subsection].append({'type': 'text', 'content': text})
            else:
                if None not in self.sections[section]:
                    self.sections[section][None] = []
                self.sections[section][None].append({'type': 'text', 'content': text})
        else:
            print(f"Section '{section}' does not exist.")

    def add_figure_to_section(self, section, figure, subtitle=None, is_plotly=False, subsection=None):
        '''
        Add a figure to a section in the report.

        param: section: str: name of the section
        param: figure: matplotlib figure or plotly figure: figure to add
        param: subtitle: str: subtitle for the figure(optional)
        param: is_plotly: bool: whether the figure is a plotly figure or not
        param: subsection: str: name of the subsection(optional)
        '''
        if section in self.sections:
            if subsection:
                if subsection not in self.sections[section]:
                    self.add_subsection(section, subsection)
                if is_plotly:
                    figure_html = pio.to_html(figure, full_html=False, include_plotlyjs='cdn')
                    self.sections[section][subsection].append({'type': 'plotly_figure', 'content': figure_html, 'subtitle': subtitle})
                else:
                    buffer = BytesIO()
                    figure.savefig(buffer, format='png')
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                    self.sections[section][subsection].append({'type': 'figure', 'content': image_base64, 'subtitle': subtitle})
            else:
                if None not in self.sections[section]:
                    self.sections[section][None] = []
                if is_plotly:
                    figure_html = pio.to_html(figure, full_html=False, include_plotlyjs='cdn')
                    self.sections[section][None].append({'type': 'plotly_figure', 'content': figure_html, 'subtitle': subtitle})
                else:
                    buffer = BytesIO()
                    figure.savefig(buffer, format='png')
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                    self.sections[section][None].append({'type': 'figure', 'content': image_base64, 'subtitle': subtitle})
        else:
            print(f"Section '{section}' does not exist.")

    def add_dataframe_to_section(self, section, dataframe, subtitle=None, subsection=None):
        '''
        Add a dataframe to a section in the report.

        param: section: str: name of the section
        param: dataframe: pandas dataframe: dataframe to add
        param: subtitle: str: subtitle for the dataframe(optional)
        param: subsection: str: name of the subsection(optional)
        '''
        if section in self.sections:
            if subsection:
                if subsection not in self.sections[section]:
                    self.add_subsection(section, subsection)
                html_table = dataframe.to_html(index=False)
                self.sections[section][subsection].append({'type': 'dataframe', 'content': html_table, 'subtitle': subtitle})
            else:
                if None not in self.sections[section]:
                    self.sections[section][None] = []
                html_table = dataframe.to_html(index=False)
                self.sections[section][None].append({'type': 'dataframe', 'content': html_table, 'subtitle': subtitle})
        else:
            print(f"Section '{section}' does not exist.")

    def format_to_html(self):
        '''
        Formats the report content to HTML.
        
        return: str: HTML content of the report
        '''
        html_content = """
        <html>
        <head>
            <title>SynthAssess Report</title>
            <style>
                body { font-family: 'Helvetica Neue', Arial, sans-serif; background-color: #f9f9f9; color: #333; margin: 0; padding: 0; }
                h1 { color: #f2f2f2; text-align: center; padding: 40px 0; }
                h2 { color: #34495e; border-bottom: 2px solid #ddd; padding-bottom: 5px; }
                p { font-size: 16px; line-height: 1.6; }
                .section { margin: 20px 0; padding: 10px; background-color: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .subsection { margin: 20px 0; padding: 10px; background-color: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                img { width: 100%; height: auto; display: block; margin: 0; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                table, th, td { border: 1px solid #ddd; }
                th, td { padding: 12px; text-align: left; }
                th { background-color: #f2f2f2; }
                .subtitle { font-size: 18px; color: #777; margin-bottom: 10px; }
                .collapsible { background-color: white; color: #2c3e50; cursor: pointer; padding: 10px; width: 100%; border: none; text-align: center; outline: none; font-size: 22px; margin-bottom: 10px; }
                .subcollapsible { background-color: white ; color: #2c3e50; cursor: pointer; padding: 10px; width: 100%; border: none; text-align: center; outline: none; font-size: 18px; margin-bottom: 10px; }
                .content { padding: 0; display: none; overflow: hidden; background-color: white; }
                .content p { margin: 10px 0; }
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script>
                document.addEventListener('DOMContentLoaded', function() {
                    var coll = document.getElementsByClassName('collapsible');
                    for (var i = 0; i < coll.length; i++) {
                        coll[i].addEventListener('click', function() {
                            this.classList.toggle('active');
                            var content = this.nextElementSibling;
                            if (content.style.display === 'block') {
                                content.style.display = 'none';
                            } else {
                                content.style.display = 'block';
                                window.dispatchEvent(new Event('resize'));
                            }
                        });
                    }

                    var subcoll = document.getElementsByClassName('subcollapsible');
                    for (var i = 0; i < subcoll.length; i++) {
                        subcoll[i].addEventListener('click', function() {
                            this.classList.toggle('active');
                            var subcontent = this.nextElementSibling;
                            if (subcontent.style.display === 'block') {
                                subcontent.style.display = 'none';
                            } else {
                                subcontent.style.display = 'block';
                                window.dispatchEvent(new Event('resize'));
                            }
                        });
                    }

                    // Trigger resize event for Plotly figures after page load
                    window.dispatchEvent(new Event('resize'));
                });
            </script>
        </head>
        <body>
            <h1 style="background-color:#4d4d4d;">SynthAssess Report</h1>
        """
        for section, subsections in self.sections.items():
            html_content += f"<div class='section'><button class='collapsible'>{section} {self.icons[section]}</button><div class='content'>"
            for subsection, contents in subsections.items():
                if subsection is not None:
                    html_content += f"<div class='subsection'><button class='subcollapsible'>{subsection}</button><div class='content'>"
                for item in contents:
                    if item['type'] == 'text':
                        html_content += f"<p>{item['content']}</p>"
                    elif item['type'] == 'figure':
                        if item['subtitle']:
                            html_content += f"<div class='subtitle'>{item['subtitle']}</div>"
                        html_content += f"<img src='data:image/png;base64,{item['content']}'/>"
                    elif item['type'] == 'plotly_figure':
                        if item['subtitle']:
                            html_content += f"<div class='subtitle'>{item['subtitle']}</div>"
                        html_content += item['content']
                    elif item['type'] == 'dataframe':
                        if item['subtitle']:
                            html_content += f"<div class='subtitle'>{item['subtitle']}</div>"
                        html_content += item['content']
                if subsection is not None:
                    html_content += "</div></div>"
            html_content += "</div></div>"

        html_content += """
        </body>
        </html>
        """
        return html_content


    def generate_report(self):
        '''
        Generates the report in HTML format.
        
        return: str: HTML content of the report'''
        html_report = self.format_to_html()
        return html_report
