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
print
"\@htmlonly
<table class=\"example\" width=\"50%\">
<tr><th colspan=\"2\"><b><small>Table of contents</small></b></th></tr>
<tr><td width=\"50%\" valign=\"top\">
<ol>
  <li> <a href=\"#${example_underscore}_Intro\" class=bold>Introduction</a>
";

system $^X, "$cmake_source_dir/doc/scripts/intro2toc", "$cmake_source_dir/examples/$example/doc/intro.dox", $example_underscore;

print "  <li> <a href=\"#${example_underscore}_CommProg\" class=bold>The commented program</a>\n";

system $^X, "$cmake_source_dir/doc/scripts/program2toc", "$cmake_source_dir/examples/$example/$example.cpp", $example_underscore;

print
"</ol></td><td width=\"50%\" valign=\"top\"><ol>
  <li value=\"3\"> <a href=\"#${example_underscore}_Results\" class=bold>Results</a>
";

system $^X, "$cmake_source_dir/doc/scripts/intro2toc", "$cmake_source_dir/examples/$example/doc/results.dox", $example_underscore;

print
"  <li> <a href=\"#${example_underscore}_PlainProg\" class=bold>The plain program</a>
</ol> </td> </tr> </table>
\@endhtmlonly
";

system $^X, "$cmake_source_dir/doc/scripts/create_anchors", "$cmake_source_dir/examples/$example/doc/intro.dox", $example_underscore;

print " * <a name=\"${example_underscore}_CommProg\"></a>\n";
print " * <h1> The commented program</h1>\n";
system $^X, "$cmake_source_dir/doc/scripts/program2doxygen", "$cmake_source_dir/examples/$example/$example.cpp", $example_underscore;

system $^X, "$cmake_source_dir/doc/scripts/create_anchors", "$cmake_source_dir/examples/$example/doc/results.dox", $example_underscore;

print
"<a name=\"${example_underscore}_PlainProg\"></a>
<h1> The plain program</h1>
\@include \"$example.cpp\"
 */
";
