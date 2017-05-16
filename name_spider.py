# -*- coding: utf-8 -*-
# @Time    : 2017/5/14 16:11
# @Author  : Studog

import urllib.request as urllib2
import lxml.html as HTML
from multiprocessing import Pool


class PersonName(object):
    def __init__(self):
        self.url = 'http://www.resgain.net/xmdq.html'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'}

    # 取所有人姓的链接
    def get_all_surname(self):
        req = urllib2.Request(self.url, headers=self.headers)
        content = urllib2.urlopen(req).read()
        content = content.decode('utf-8')
        html = HTML.fromstring(content)
        surname_url = html.xpath("//div[@class='col-xs-12']/a/@href")
        return surname_url

    # 取每个姓对应的所有姓名
    def get_all_name(self, name_url):
        name = urllib2.Request(name_url, headers=self.headers)
        name_content = urllib2.urlopen(name).read()
        name_content = name_content.decode('utf-8')
        name_html = HTML.fromstring(name_content)
        name_list = name_html.xpath("//div[@class='col-xs-12']/a/text()")
        return name_list

    # 取每个姓对应男性和女性姓名的所有页
    def find_all_page(self, page_url):
        page = urllib2.Request(page_url, headers=self.headers)
        page_content = urllib2.urlopen(page).read()
        page_content = page_content.decode('utf-8')
        page_html = HTML.fromstring(page_content)
        page_list = page_html.xpath("//ul[@class='pagination']/li/a[@class='mhidden']/text()")
        return page_list

    # 取最终需要爬取姓名的链接
    def get_final_url(self, final_url):
        final_page = [final_url[:-5] + '_%s.html' % p for p in self.find_all_page(final_url)]
        return final_page

    # 爬取所有姓名并写入txt文件
    def get_person_name(self, surname):
        boy_url = surname[:-14] + 'name/boys.html'
        girl_url = surname[:-14] + 'name/girls.html'
        with open(r'E:\DataSets\person.txt', 'a', encoding='utf-8') as f:
            for b_page in self.get_final_url(boy_url):
                boys = self.get_all_name(b_page)
                for boy in boys:
                    f.write(boy + ',男')
                    f.write('\n')
            for g_page in self.get_final_url(girl_url):
                girls = self.get_all_name(g_page)
                for girl in girls:
                    f.write(girl + ',女')
                    f.write('\n')

    # 开启多进程
    def multi_process(self):
        pool = Pool()
        # for surname in self.get_all_surname():
        #     pool.apply_async(self.get_person_name, (surname,))
        pool.map(self.get_person_name, self.get_all_surname())
        pool.close()
        pool.join()


if __name__ == '__main__':
    person = PersonName()
    person.multi_process()