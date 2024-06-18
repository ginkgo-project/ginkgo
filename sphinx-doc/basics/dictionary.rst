.. _basics_dictionary:

Dictionary
==========

The Dictionary class is a tree-like data structure that can store any type of data similar to json, yaml or toml format. It is used to store the input data for the simulation. The Dictionary class is a wrapper around std::unordered_map<std::string, std::any> and provides a simple interface to insert, retrieve and modify values.

This generic interface allows to deliver complex input data to the simulation code without the need to define a specific data structure. It can be used as followed (more details see test_Dictionary.cpp):

.. code-block:: cpp

    NeoFOAM::Dictionary dict;

    dict.insert("key1", 42);
    dict.insert("key2", std::string("Hello"));

    dict.get<int>("key1") == 42;
    dict.get<std::string>("key2") == "Hello";

If the dictionary is not passed as a const reference, the values can be modified as well:

.. code-block:: cpp

    NeoFOAM::Dictionary dict;
    dict.insert("key", 42);
    dict["key"] = 43;

    dict.get<int>("key") == 43;

Accessing a non-existent key will throw an exception std::out_of_range:

.. code-block:: cpp

    NeoFOAM::Dictionary dict;
    dict["non_existent_key"]; // will throw with std::out_of_range;

To check if a key exists, the found method can be used:

.. code-block:: cpp

    NeoFOAM::Dictionary dict;
    dict.insert("key", 42);
    dict.found("key") == true;

Dictionary provides a method to remove a key:

.. code-block:: cpp

    NeoFOAM::Dictionary dict;
    dict.insert("key", 42);
    dict["key"] = 43;
    dict.remove("key");

    dict.found("key") == false;

The Dictionary class also provides a method to access a sub-dictionary. This is useful to group related data together:

.. code-block:: cpp

    NeoFOAM::Dictionary dict;
    NeoFOAM::Dictionary subDict;
    subDict.insert("key1", 42);
    subDict.insert("key2", std::string("Hello"));

    dict.insert("subDict", subDict);

    NeoFOAM::Dictionary& sDict = dict.subDict("subDict");
    sDict.get<int>("key1") == 42;
    sDict.get<std::string>("key2") == "Hello";

    sDict.get<int>("key1") = 100;

    // check if the value is modified
    NeoFOAM::Dictionary& sDict2 = dict.subDict("subDict");
    sDict2.get<int>("key1") == 100;

.. doxygenclass:: NeoFOAM::Dictionary
   :members:
