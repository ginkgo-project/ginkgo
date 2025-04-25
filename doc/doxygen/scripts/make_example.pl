## ---------------------------------------------------------------------
##
## Copyright (C) 2013 - 2019 by the deal.II authors
##
## The original file is from the deal.II library. It has been modified for
## Ginkgo.
##
## The deal.II library is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
## The full text of the license can be found in the file LICENSE.md at
## the top level directory of deal.II.
##
## ---------------------------------------------------------------------


if ($#ARGV != 2) {
  print "\nUsage: make_example.pl example cmake_source_dir cmake_binary_dir\n";
  exit;
}

$example=$ARGV[0];
$example_underscore=$example;
$example_underscore=~ s/-/_/g;

$cmake_source_dir=$ARGV[1];
$cmake_build_dir=$ARGV[2];

print
"/** \@page $example_underscore The $example program
";

open intro, "$cmake_source_dir/examples/$example/doc/short-intro"
    or die "Can't open builds-on file $cmake_source_dir/examples/$example/doc/short-intro";
my $shortintro= <intro>;
close intro;
chop $shortintro;

if ($shortintro ne "")
{
    print "$shortintro.\n\n";
}

open BF, "$cmake_source_dir/examples/$example/doc/builds-on"
    or die "Can't open builds-on file $cmake_source_dir/examples/$example/doc/builds-on";
my $buildson = <BF>;
close BF;
chop $buildson;

# At the very top, print which other programs this one builds on.
if ($buildson ne "")
{
    $buildson =~ s/ /, /g;
    print "This example depends on $buildson.\n\n";
}

# then show the table of contents
print "\@tableofcontents\n";

open(my $intro, '<', "$cmake_source_dir/examples/$example/doc/intro.dox")
    or die "Can't open intro file $cmake_source_dir/examples/$example/doc/intro.dox";
while(<$intro>){ print ;}
close($intro);

# generating a unique id for section names, curtesy of claude
my @chars = ('a'..'z');
my $sec1 = join '', map { $chars[rand @chars] } 1..6;
my $sec2 = join '', map { $chars[rand @chars] } 1..6;

print " * \@section $sec1 The commented program\n";
system $^X, "$cmake_source_dir/doc/doxygen/scripts/program2doxygen", "$cmake_source_dir/examples/$example/$example.cpp";

open(my $results, '<', "$cmake_source_dir/examples/$example/doc/results.dox")
    or die "Can't open results file $cmake_source_dir/examples/$example/doc/results.dox";
while(<$results>){ print ;}
close($results);

print "\@section $sec2 The plain program\n";
print " * \@code\n";
open(my $prog, '<', "$example.cpp") or die "Can't open plain program $example.cpp";
while(my $line = readline($prog)){
    print " *    $line"
}
close $prog;
print " * \@endcode\n";
print "*/";
