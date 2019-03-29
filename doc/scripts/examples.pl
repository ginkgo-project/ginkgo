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

use strict;

my $example_file = shift;
open EXAMPLE, "<$example_file";

# Print the first part of example.hpp.in up until the point where we
# find the line with '@@EXAMPLE_MAP@@'
while (my $line = <EXAMPLE>)
{
  last if($line =~ m/\@\@EXAMPLE_MAP\@\@/);
  print $line;
}

# List of additional node attributes to highlight purpose and state of the example
my %style = (
    "basic"          => ',height=.8,width=.8,shape="octagon",fillcolor="forestgreen",peripheries=3',
    "techniques"     => ',height=.35,width=.35,fillcolor="coral"',
    "logging"         => ',height=.25,width=.25,fillcolor="gold"',
    "stopping-criteria"         => ',height=.25,width=.25,fillcolor="deepskyblue"',
    "preconditioners" => ',height=.25,width=.25,fillcolor="crimson"',
    "unfinished"     => ',height=.25,width=.25,style="dashed"',
    );

# Print a preamble setting common attributes
print << 'EOT'
digraph ExamplesMap
{
  overlap=false;
  edge [fontname="Helvetica",
        fontsize="13",
        labelfontname="Helvetica",
        labelfontsize="12",
        color="black",
        style="solid"];
  node [fontname="Helvetica",
        fontsize="12",
        shape="rectangle",
        height=0.2,
        width=0.4,
        color="black",
        fillcolor="white",
        style="filled"];
EOT
    ;

# Print all nodes of the graph by looping over the remaining
# command line arguments denoting the example programs

my $example;
foreach $example (@ARGV)
{
    # read first line of tooltip file
    open TF, "$example/doc/tooltip"
        or die "Can't open tooltip file $example/doc/tooltip";
    my $tooltip = <TF>;
    close TF;
    chop $tooltip;

    # read first line of 'kind' file if it is a example;
    # otherwise assume it is a code gallery program. for
    # each of them, output something for 'dot' to generate
    # the dependencies graph from
    if (!($example =~ /code-gallery/))
    {
      open KF, "$example/doc/kind"
          or die "Can't open kind file $example/doc/kind";
      my $kind = <KF>;
      chop $kind;
      close KF;

      die "Unknown kind '$kind' in file $example/doc/kind" if (! defined $style{$kind});

      my $name = $example;
      $name =~ s/^.*examples\///;
      my $tag = $name;
      $tag=~ s/-/_/g;

      printf "  $tag [label=\"$tag\", URL=\"\\ref $tag\", tooltip=\"$tooltip\"";
      print "$style{$kind}";
    }
    else
    {
      # get at the name of the program; also create something
      # that can serve as a tag without using special characters
      my $name = $example;
      $name =~ s/^.*code-gallery\///;
      my $tag = $name;
      $tag =~ s/[^a-zA-Z]/_/g;

      printf "  code_gallery_$tag [label=\"\", URL=\"\\ref code_gallery_$tag\", tooltip=\"$tooltip\"";
      my $kind = "code-gallery";
      print "$style{$kind}";
    }

    print "];\n";
}

# Print all edges by going over the same list of examples again.
# Keep sorted by second node on edge!

my $example;
foreach $example (@ARGV)
{
    # read first line of dependency file
    open BF, "$example/doc/builds-on"
        or die "Can't open builds-on file $example/doc/builds-on";
    my $buildson = <BF>;
    close BF;
    chop $buildson;

    my $destination;
    if (!($example =~ /code-gallery/))
    {
        my $name = $example;
        $name =~ s/^.*examples\///;
        my $tag = $name;
        $tag=~ s/-/_/g;
        $destination = "$tag";
    }
    else
    {
      my $name = $example;
      $name =~ s/^.*code-gallery\///;
      my $tag = $name;
      $tag =~ s/[^a-zA-Z]/_/g;
      $destination = "code_gallery_$tag";
    }

    my $source;
    foreach $source (split ' ', $buildson) {
        $source =~ $example;
        $source =~ s/^.*examples\///;
        $source =~ s/-/_/g;
        print "  $source -> $destination";
        if ($destination =~ /code_gallery/)
        {
            print " [style=\"dashed\", arrowhead=\"empty\"]";
        }
        print "\n";
    }
}

print "}\n";

# Copy that part of example.hpp.in up until the point where we
# find the line with '@@EXAMPLE_LEGEND@@'
while (my $line = <EXAMPLE>)
{
  last if($line =~ m/\@\@EXAMPLE_LEGEND\@\@/);
  print $line;
}

# Print a preamble setting common attributes
print << 'EOT'
graph ExamplesDescription
{
  overlap=false;
  edge [fontname="Helvetica",
        fontsize="10",
        labelfontname="Helvetica",
        labelfontsize="10",
        color="black",
        style="solid"];
  node [fontname="Helvetica",
        fontsize="10",
        shape="rectangle",
        height=0.2,
        width=0.4,
        color="black",
        fillcolor="white",
        style="filled"];
EOT
    ;

my %kind_descriptions = (
    "basic"          => 'Basic techniques',
    "techniques"     => 'Advanced techniques',
    "logging"         => 'Logging in Ginkgo',
    "stopping-criteria"         => 'Stopping criteria',
    "preconditioners" => 'Preconditioners',
    "unfinished"     => 'Unfinished codes',
    );

# for each kind, print a box in the same style as used in
# the connections graph; also print a fake box with a
# description of what each kind is. then connect these
my $kind;
foreach $kind (keys %style)
{
    my $escaped_kind = $kind;
    $escaped_kind =~ s/[^a-zA-Z]/_/g;
    printf "  $escaped_kind [label=\"\" $style{$kind}];\n";
    printf "  fake_$escaped_kind [label=\"$kind_descriptions{$kind}\", shape=plaintext];\n";
    printf "  $escaped_kind -- fake_$escaped_kind [style=dotted, arrowhead=odot, arrowsize=1];\n";
}
# now add connections to make sure they appear nicely next to each other
# in the legend
print "  basic -- techniques -- logging -- stopping_criteria -- preconditioners -- unfinished;\n";

# we need to tell 'dot' that all of these are at the same
# rank to ensure they appear next to (as opposed to atop)
# each other
print "  {rank=same; basic, techniques, logging, stopping_criteria, preconditioners, unfinished}";

# end the graph
print "}\n";



# Then print the rest of example.hpp.in
while (my $line = <EXAMPLE>)
{
  print $line;
}
close EXAMPLE;
