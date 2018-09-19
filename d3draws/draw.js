"use strict";

class IntuitionPlot {
    constructor(svg_id) {
        this.female_color = "blue";
        this.male_color = "green";
        this.data = [{"x": 143.00769000291828, "y": 76.26041945593573, "color": this.female_color},
                    {"x": 143.6771726099216, "y": 48.36603303425654, "color": this.female_color},
                    {"x": 144.2722682605912, "y": 85.07996810396665, "color": this.female_color},
                    {"x": 144.2722682605912, "y": 65.80002454780602, "color": this.female_color},
                    {"x": 145.53684651826418, "y": 53.493677597065215, "color": this.female_color},
                    {"x": 147.47090738294048, "y": 71.13277489312705, "color": this.female_color},
                    {"x": 147.47259595113408, "y": 81.58851394332488, "color": this.female_color},
                    {"x": 149.40649705599813, "y": 62.28817126498637, "color": this.female_color},
                    {"x": 150.00006389828638, "y": 46.520080991645415, "color": this.female_color},
                    {"x": 173.07172176031852, "y": 3.9866460256924796, "color": this.male_color},
                    {"x": 175.05387799980326, "y": 3.2750709816425, "color": this.male_color},
                    {"x": 179.01408450704224, "y": 6.407766990291265, "color": this.male_color},
                    {"x": 180.00000000000000, "y": 5.242718446601941, "color": this.male_color},
                    {"x": 185.04802173004478, "y": 3.9866460256924796, "color": this.male_color},
                     {"x": 190.01140778078064, "y": 6.09713881875949, "color": this.male_color}];
        this.margin = {top: 20, right: 20, bottom: 60, left: 65};
        this.width = 600;
        this.height = 400;

        // Setup x and y scales
        this.xScale = d3.scaleLinear().domain([140, 200]).range([0, this.width]);
        this.yScale = d3.scaleLinear().domain([0, 100]).range([this.height, 0]);

        // Save the ID
        this.svg_id = svg_id;
        this.svg_group = d3.select(svg_id)
        // .append("svg")
        // .selectAll("*").remove()
            .attr("viewBox", "0 0 " + this.total_width + " " + this.total_height)
            .append("g")
            .attr("transform", "translate(" + this.margin.left + "," + this.margin.top + ")");

        this.draw();
        // this.add_first_separating_line();
        // this.add_second_separating_line();
        // this.add_third_separating_line();
    }

    draw() {
        // Clean any previous content such that we can call draw multiple times
        // and the content will be replaced instead of duplicated
        this.svg_group.selectAll("*").remove();

        this.addAxis();
        this.addData();
    }

    get total_width() {
        return this.width + this.margin.left + this.margin.right;;
    }

    get total_height() {
        return this.height + this.margin.top + this.margin.bottom;
    }

    addAxis() {
        var xAxis = d3.axisBottom().scale(this.xScale);
        var yAxis = d3.axisLeft().scale(this.yScale);

        var xGroup = this.svg_group.append("g");
        xGroup.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + this.height + ")")
            .call(xAxis);

        xGroup.append("text")
            .attr("transform",
                  "translate(" + (this.width/2) + " ," + (this.height + 0.9*this.margin.bottom) + ")")
            .style("text-anchor", "middle")
            .text("Height");

        var yGroup = this.svg_group.append("g");
        yGroup.append("g")
            .attr("class", "y axis")
            .call(yAxis);

        yGroup.append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", 0 - this.margin.left)
            .attr("x",0 - (this.height / 2))
            .attr("dy", "1em")
            .style("text-anchor", "middle")
            .text("Hair Length (cm)");
    }

    addData() {
        var xScale = this.xScale;
        var yScale = this.yScale;
        var xValue = function(d) { return d.x; };
        var xMap = function(d) { return xScale(xValue(d)); };
        var yValue = function(d) { return d.y; };
        var yMap = function(d) { return yScale(yValue(d)); };
        var colorValue = function(d) { return d.color; };

        this.svg_group.append("g")
            .selectAll("circle")
            .data(this.data)
            .enter()
            .append("circle")
            .attr("cx", xMap)
            .attr("cy", yMap)
            .attr("r", 5)
            .attr("stroke", "black")
            .attr("fill",  colorValue);
    }

    add_first_separating_line() {
        this.svg_group.append("line")
            .attr("x1", this.xScale(144))
            .attr("y1", this.yScale(1))
            .attr("x2", this.xScale(164))
            .attr("y2", this.yScale(90))
            .style("stroke", "rgb(255,0,0)")
            .style("stroke-width", 2);
    }

    add_second_separating_line() {
        this.svg_group.append("line")
            .attr("x1", this.xScale(155))
            .attr("y1", this.yScale(1))
            .attr("x2", this.xScale(194))
            .attr("y2", this.yScale(20))
            .style("stroke", "rgb(255,0,0)")
            .style("stroke-width", 2);
    }

    add_third_separating_line() {
        this.svg_group.append("line")
            .attr("x1", this.xScale(150))
            .attr("y1", this.yScale(1))
            .attr("x2", this.xScale(180))
            .attr("y2", this.yScale(90))
            .style("stroke", "rgb(255,0,0)")
            .style("stroke-width", 2);
    }
}
