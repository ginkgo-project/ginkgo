#!/bin/bash
sed -e :a -e '/\\$/N;s/\\\n */ /;ta' $1 | sed '/#/d;/^\w*$/d' | sort
