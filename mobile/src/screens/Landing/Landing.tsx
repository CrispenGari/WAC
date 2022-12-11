import { View, Text, TouchableOpacity, StatusBar, Image } from "react-native";
import React from "react";
import { COLORS, FONTS } from "../../constants";

const Landing = ({ navigation }) => {
  return (
    <View
      style={{
        flex: 1,
        padding: 10,
        backgroundColor: COLORS.main,
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <StatusBar barStyle={"light-content"} />
      <View
        style={{
          alignItems: "center",
          marginBottom: 30,
          flex: 0.7,
          justifyContent: "center",
        }}
      >
        <Text
          style={{
            fontFamily: FONTS.NunitoSansExtraBold,
            color: "white",
            letterSpacing: 3,
            fontSize: 30,
            marginBottom: 30,
          }}
        >
          WAC Tool
        </Text>
        <Image
          source={{
            uri: Image.resolveAssetSource(require("../../../assets/logo.png"))
              .uri,
          }}
          style={{
            width: 100,
            height: 100,
            marginBottom: 30,
          }}
        />
        <Text
          style={{
            textAlign: "center",
            color: "white",
            fontFamily: FONTS.NunitoSansRegular,
          }}
        >
          Wild Animal Classifier (WAC) is a mobile AI tool for basic image
          classification based on 6 animals from wild images in real time. (If
          you don't like wild animals don't open this app.)
        </Text>
      </View>
      <View
        style={{
          justifyContent: "space-between",
          flex: 0.3,
          width: "100%",
          alignItems: "center",
        }}
      >
        <TouchableOpacity
          onPress={() => {
            navigation.replace("Home", {});
          }}
          style={{
            marginVertical: 30,
            backgroundColor: COLORS.lime,
            width: "90%",
            paddingHorizontal: 20,
            paddingVertical: 15,
            justifyContent: "center",
            alignItems: "center",
            borderRadius: 999,
          }}
          activeOpacity={0.7}
        >
          <Text
            style={{
              color: "white",
              fontFamily: FONTS.NunitoSansRegular,
              letterSpacing: 2,
              fontSize: 20,
            }}
          >
            open tool
          </Text>
        </TouchableOpacity>
        <Text
          style={{
            margin: 20,
            color: "white",
            textAlign: "center",
            fontFamily: FONTS.NunitoSansItalic,
          }}
        >
          AI tool developed by @crispengari.
        </Text>
      </View>
    </View>
  );
};

export default Landing;
